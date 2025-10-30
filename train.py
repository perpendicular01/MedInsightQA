from conf.args import Arguments
from model import MCQAModel
from dataset import MCQADataset
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pytorch_lightning as pl
import torch, os
from tqdm import tqdm
import argparse
import time
import json

EXPERIMENT_DATASET_FOLDER = "/content/"
WB_PROJECT = "MEDMCQA"


def train(gpu, args, exp_dataset_folder, experiment_name, models_folder, version):
    pl.seed_everything(42)

    EXPERIMENT_FOLDER = os.path.join(models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

    # Loggers
    wb = WandbLogger(project=WB_PROJECT, name=experiment_name, version=version)
    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    # Datasets
    train_dataset = MCQADataset(os.path.join(exp_dataset_folder, "train.json"), args.use_context)
    val_dataset = MCQADataset(os.path.join(exp_dataset_folder, "dev.json"), args.use_context)
    test_dataset_path = os.path.join(exp_dataset_folder, "test.json")
    if os.path.exists(test_dataset_path):
        test_dataset = MCQADataset(test_dataset_path, args.use_context)
    else:
        test_dataset = None

    # Callbacks
    es_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=2,
        verbose=True,
        mode='min'
    )

    cp_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=EXPERIMENT_FOLDER,
        filename=experiment_name + "-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=1,
        save_weights_only=True,
        mode='min'
    )

    # Model
    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                          args=args.__dict__)
    mcqaModel.prepare_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    # Trainer (Lightning 2.x)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        logger=[wb, csv_log],
        callbacks=[es_callback, cp_callback],
        max_epochs=args.num_epochs,
        log_every_n_steps=10,
        enable_checkpointing=True
    )

    # Train
    trainer.fit(mcqaModel)
    print(f"Training completed")

    # Load best checkpoint for testing
    ckpt_path = cp_callback.best_model_path
    inference_model = MCQAModel.load_from_checkpoint(ckpt_path)

    # Assign test dataset to the new instance
    inference_model.prepare_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    
    # Save predictions for validation set
    val_dataset_list = load_jsonlines(os.path.join(exp_dataset_folder, "dev.json"))
    val_predictions = run_inference(inference_model, inference_model.val_dataloader())
    for i, pred in enumerate(val_predictions):
        val_dataset_list[i]['prediction'] = int(pred + 1)
    save_json(val_dataset_list, os.path.join(EXPERIMENT_FOLDER, "dev_results.json"))
    print(f"Validation predictions written to {os.path.join(EXPERIMENT_FOLDER,'dev_results.json')}")

    # Cleanup
    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="name of the model")
    parser.add_argument("--dataset_folder_name", default="dataset", help="dataset folder containing train.json, dev.json, test.json")
    parser.add_argument("--use_context", default=False, action='store_true', help="use context flag")
    cmd_args = parser.parse_args()

    exp_dataset_folder = os.path.join(EXPERIMENT_DATASET_FOLDER, cmd_args.dataset_folder_name)
    model_name = cmd_args.model
    print(f"Training started for model {model_name}, dataset {exp_dataset_folder}, use_context={cmd_args.use_context}")

    args = Arguments(
        train_csv=os.path.join(exp_dataset_folder, "train.json"),
        test_csv=os.path.join(exp_dataset_folder, "test.json"),
        dev_csv=os.path.join(exp_dataset_folder, "dev.json"),
        pretrained_model_name=model_name,
        use_context=cmd_args.use_context
    )

    exp_name = f"{model_name}@@@{os.path.basename(exp_dataset_folder)}@@@use_context{str(cmd_args.use_context)}@@@seqlen{args.max_len}".replace("/", "_")

    train(
        gpu=args.gpu,
        args=args,
        exp_dataset_folder=exp_dataset_folder,
        experiment_name=exp_name,
        models_folder="./models",
        version=exp_name
    )

    time.sleep(60)

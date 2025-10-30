from torch.utils.data import Dataset
import json

class MCQADataset(Dataset):
    """
    Multiple Choice QA Dataset
    Supports:
        - Context (from 'exp') if use_context=True
        - Labels ('cop') for train/dev
        - No labels for test dataset (label=None)
    """

    def __init__(self, json_path, use_context=True):
        self.use_context = use_context
        self.dataset = []

        # Read JSON Lines file
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return_tuple = ()

        # Include context if requested
        if self.use_context:
            context = item.get('exp', None)  # 'exp' for explanation/context if exists
            return_tuple += (context,)

        # Question and options
        question = item['question']
        options = [item['opa'], item['opb'], item['opc'], item['opd']]

        # Always include label; set None for test
        label = item.get('cop', None)
        if label is not None:
            label = label - 1  # Convert 1-indexed to 0-indexed

        return_tuple += (question, options, label)  # Always 3 elements

        return return_tuple

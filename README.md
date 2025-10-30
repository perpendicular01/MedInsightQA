# MedInsightQA: Exploring Underrepresented Medical Topics using Transformer Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)

> A novel medical question-answering benchmark focusing on underrepresented medical specialties to enable comprehensive evaluation of transformer models in specialized domains.

## 📖 Overview

MedInsightQA addresses the critical gap in medical AI evaluation by introducing a carefully curated dataset focusing on **Neurology**, **First Aid**, and **Respiratory Medicine**. Through comprehensive evaluation of both general-domain and medical-specific transformer models, this project challenges conventional assumptions about domain-specific pre-training superiority.

### Key Contributions

- 🏥 **MedInsightQA Dataset**: 300 carefully curated multiple-choice questions across underrepresented medical topics
- 🤖 **Comprehensive Evaluation**: Comparison of 6 transformer models (2 general-domain + 4 medical-specific)
- 📊 **Performance Analysis**: Zero-shot vs. fine-tuned performance evaluation
- 🔬 **Novel Insights**: Empirical evidence on the effectiveness of domain-specific pre-training


## 🗂️ Dataset

### Structure

The MedInsightQA dataset contains 300 multiple-choice questions with the following format:

```json
{
  "id": "unique_identifier",
  "question": "Question text?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "Correct option letter (A/B/C/D)",
  "explanation": "Brief explanation for the correct answer"
}
```

### Dataset Distribution

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 240 | 80% |
| Validation | 30 | 10% |
| Test | 30 | 10% |

### Topics Covered

- 🧠 **Neurology**: Disorders of the nervous system
- 🚑 **First Aid**: Emergency medical response
- 🫁 **Respiratory Medicine**: Respiratory conditions and treatments

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- pip or conda package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/medinsightqa.git
cd medinsightqa

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=1.12.0
transformers>=4.20.0
pytorch-lightning>=1.7.0
datasets>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### Zero-Shot Evaluation

```python
from evaluation import ZeroShotEvaluator

# Initialize evaluator
evaluator = ZeroShotEvaluator()

# Evaluate all models on test set
results = evaluator.evaluate_all_models(test_loader)
print(results)
```

### Fine-Tuning Models

```python
from training import MedQATrainer

# Initialize trainer with desired model
trainer = MedQATrainer(model_name='bert-base-uncased')

# Train the model
trainer.fit(train_loader, val_loader)

# Evaluate on test set
test_accuracy = trainer.evaluate(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

## 🤖 Models Evaluated

### General-Domain Models

- `bert-base-uncased` - Google's BERT base model
- `roberta-base` - Facebook's RoBERTa base model

### Medical-Specific Models

- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` - BiomedBERT
- `NeuML/pubmedbert-base-embeddings` - PubMedBERT
- `emilyalsentzer/Bio_ClinicalBERT` - Bio_ClinicalBERT
- `dmis-lab/biobert-v1.1` - BioBERT v1.1

## 📊 Results

### Performance Summary

| Model | Zero-Shot Accuracy | Fine-Tuned Accuracy | Improvement |
|-------|-------------------|--------------------|--------------------|
| **BERT Base Uncased** | 33.33% | **56.67%** | +23.33% |
| **RoBERTa Base** | 30.00% | 53.33% | +23.33% |
| **BiomedBERT** | 26.67% | 46.67% | +20.00% |
| **Bio_ClinicalBERT** | 36.67% | 50.00% | +13.33% |
| **PubMedBERT** | 13.33% | 50.00% | **+36.67%** |
| **BioBERT-v1.1** | 30.00% | 40.00% | +10.00% |

### Key Findings

✅ Fine-tuning significantly improves performance across all models  
✅ BERT-base achieved highest accuracy (56.67%) after fine-tuning  
✅ PubMedBERT showed most dramatic improvement (+36.67%) despite low zero-shot performance  
✅ Medical pre-training provides strong baseline but doesn't guarantee best fine-tuned performance  
✅ Domain-specific pre-training advantages can be overcome with task-specific fine-tuning


## 🔮 Future Work

- 📈 Expand dataset size with additional questions and medical topics (FCPS questions)
- 🌐 Multilingual support for Bangla-English mixed medical questions
- 🏥 Include more medical specialties for comprehensive coverage
- 🤖 Evaluate larger generative models (GPT-4, Claude, etc.)
- 🚀 Develop real-time medical QA application for healthcare practitioners



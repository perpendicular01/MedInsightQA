# MedInsightQA: Exploring Underrepresented Medical Topics using Transformer Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)

## 📖 Overview

MedInsightQA is a novel medical question-answering benchmark designed to address the underrepresentation of critical medical specialties in existing evaluation datasets. This project introduces a custom dataset focusing on **Neurology**, **First Aid**, and **Respiratory medicine** to enable more comprehensive evaluation of transformer models in specialized medical domains.

**Key Contributions:**
- 🏥 **MedInsightQA Dataset**: 300 carefully curated multiple-choice questions across underrepresented medical topics
- 🤖 **Comprehensive Evaluation**: Comparison of 6 transformer models (2 general-domain + 4 medical-specific)
- 📊 **Performance Analysis**: Zero-shot vs. fine-tuned performance evaluation
- 🔬 **Novel Insights**: Challenging assumptions about domain-specific pre-training superiority


## 🗂️ Dataset

### MedInsightQA Dataset Structure
The dataset contains 300 multiple-choice questions with the following structure:
```json
{
  "id": "unique_identifier",
  "question": "Question text?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "Correct option letter (A/B/C/D)",
  "explanation": "Brief explanation for the correct answer"
}

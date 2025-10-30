import json
import matplotlib.pyplot as plt
import numpy as np

# File paths - update these with your actual file names
answer_file_path = '/content/test_result.json'  # File with correct answers (has 'cop')
before_train_file_path = '/content/biobert without test_results.json'  # Predictions before training
after_train_file_path = '/content/biobert with test_results.json'  # Predictions after training

# Model name for the graph title
model_name = "BioBert-v1.1"

def load_predictions(file_path):
    """Load predictions from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_answers(file_path):
    """Load answers from JSON Lines file"""
    answers = []
    with open(file_path, 'r') as f:
        for line in f:
            answers.append(json.loads(line.strip()))
    return answers

def calculate_accuracy(predictions_data, answers_data, prediction_key='prediction'):
    """Calculate accuracy between predictions and answers"""
    # Create dictionary for answers
    answers_dict = {item['id']: item['cop'] for item in answers_data}

    correct = 0
    total = 0
    results = []

    for item in predictions_data:
        question_id = item['id']
        prediction = item[prediction_key]  # Use the specified key

        if question_id in answers_dict:
            actual_answer = answers_dict[question_id]
            is_correct = prediction == actual_answer

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'id': question_id,
                'prediction': prediction,
                'actual': actual_answer,
                'correct': is_correct
            })
        else:
            print(f"Warning: ID {question_id} not found in answer file")

    return correct, total, results

# Load data
answers = load_answers(answer_file_path)
before_train_predictions = load_predictions(before_train_file_path)
after_train_predictions = load_predictions(after_train_file_path)

# Calculate accuracy for before training
print("=== BEFORE TRAINING RESULTS ===")
correct_before, total_before, results_before = calculate_accuracy(before_train_predictions, answers, 'prediction')

if total_before > 0:
    accuracy_before = (correct_before / total_before) * 100
    print(f"Correct predictions: {correct_before}")
    print(f"Total questions: {total_before}")
    print(f"Accuracy: {accuracy_before:.2f}%")
else:
    print("No matching questions found for before training")

print("\n" + "="*50 + "\n")

# Calculate accuracy for after training
print("=== AFTER TRAINING RESULTS ===")
correct_after, total_after, results_after = calculate_accuracy(after_train_predictions, answers, 'prediction')

if total_after > 0:
    accuracy_after = (correct_after / total_after) * 100
    print(f"Correct predictions: {correct_after}")
    print(f"Total questions: {total_after}")
    print(f"Accuracy: {accuracy_after:.2f}%")
else:
    print("No matching questions found for after training")

print("\n" + "="*50 + "\n")


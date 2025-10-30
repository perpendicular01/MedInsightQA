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

# Comparison summary
print("=== COMPARISON SUMMARY ===")
if total_before > 0 and total_after > 0:
    accuracy_improvement = accuracy_after - accuracy_before
    correct_improvement = correct_after - correct_before

    print(f"Accuracy Before Training: {accuracy_before:.2f}%")
    print(f"Accuracy After Training:  {accuracy_after:.2f}%")
    print(f"Accuracy Improvement:     {accuracy_improvement:+.2f}%")
    print(f"Correct Answers Improvement: {correct_improvement:+d} questions")

    # Show some examples of changes
    print(f"\nSample changes (first 5 where predictions changed):")
    changes_shown = 0
    for i, (before, after) in enumerate(zip(results_before, results_after)):
        if before['prediction'] != after['prediction'] and changes_shown < 5:
            before_status = "✓" if before['correct'] else "✗"
            after_status = "✓" if after['correct'] else "✗"
            print(f"ID: {before['id']}")
            print(f"  Before: {before['prediction']} {before_status} -> After: {after['prediction']} {after_status}")
            print(f"  Actual: {before['actual']}")
            changes_shown += 1
            print()

elif total_before == 0:
    print("No valid results for before training")
elif total_after == 0:
    print("No valid results for after training")

# Create visualization
if total_before > 0 and total_after > 0:
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Data for plotting
    categories = ['Before Training', 'After Training']
    accuracies = [accuracy_before, accuracy_after]
    colors = ['#b8b8b8', '#1a80bb']

    # Create bar plot
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize the plot
    plt.title(f'Model Performance: {model_name}\n',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, max(accuracies) + 15)  # Add some space at the top

    # Add improvement arrow
    improvement_y = max(accuracies) + 8
    plt.annotate('', xy=(1, improvement_y), xytext=(0, improvement_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    plt.text(0.5, improvement_y + 1, f'Improvement: {accuracy_improvement:+.2f}%',
             ha='center', va='bottom', fontsize=11, color='red', fontweight='bold')

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Additional metrics visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart for correct/incorrect distribution (After Training)
    correct_incorrect = [correct_after, total_after - correct_after]
    labels = ['Correct', 'Incorrect']
    colors_pie = ['#b8b8b8', '#1a80bb']

    ax1.pie(correct_incorrect, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=(0.1, 0))
    ax1.set_title('After Training: Prediction Distribution', fontweight='bold')

    # Improvement visualization
    improvement_data = [correct_before, correct_improvement] if correct_improvement > 0 else [correct_after, -correct_improvement]
    improvement_labels = ['Baseline Correct', 'Improvement'] if correct_improvement > 0 else ['Final Correct', 'Decline']
    improvement_colors = ['#1a80bb', '#b8b8b8'] if correct_improvement > 0 else ['#b8b8b8', '#1a80bb']

    ax2.bar(improvement_labels, improvement_data, color=improvement_colors, alpha=0.8)
    ax2.set_title('Performance Change', fontweight='bold')
    ax2.set_ylabel('Number of Questions')

    # Add value labels on bars
    for i, (label, value) in enumerate(zip(improvement_labels, improvement_data)):
        ax2.text(i, value + (max(improvement_data)*0.01), f'{value}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print final summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Final Accuracy: {accuracy_after:.2f}%")
    print(f"Improvement: {accuracy_improvement:+.2f}%")
    print(f"Relative Improvement: {(accuracy_improvement/accuracy_before)*100:+.2f}%")
    print(f"Questions Correctly Predicted: {correct_after}/{total_after}")
import matplotlib.pyplot as plt
import numpy as np

# Your model comparison data
model_data = [
    {"Rank": 1, "Model": "BERT Base", "Accuracy": 56.67},
    {"Rank": 2, "Model": "RoBERTa Base", "Accuracy": 53.33},
    {"Rank": 3, "Model": "BiomedBERT", "Accuracy": 46.67},
    {"Rank": 4, "Model": "PubMedBERT", "Accuracy": 50.0},
    {"Rank": 5, "Model": "Bio_ClinicalBERT", "Accuracy": 50.0},
    {"Rank": 6, "Model": "biobert-v1.1", "Accuracy": 40.0}
]

# Extract data for plotting
model_names = [item["Model"] for item in model_data]
accuracies = [item["Accuracy"] for item in model_data]

# Find the best model (highest accuracy)
best_accuracy = max(accuracies)
best_model_index = accuracies.index(best_accuracy)
best_model_name = model_names[best_model_index]

# Create vertical bar chart
plt.figure(figsize=(12, 6))

# Create colors - red for best model, blue for others
colors = []
for i, acc in enumerate(accuracies):
    if acc == best_accuracy:
        colors.append('red')  # Red for best model
    else:
        colors.append('skyblue')  # Blue for other models

bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='grey', linewidth=0.5)

# Add value labels on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

# Customize the plot
plt.title('Model Accuracy before Training', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, max(accuracies) + 10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add light grid
plt.grid(axis='y', alpha=0.2, linestyle='--')
plt.gca().set_axisbelow(True)

# Clean up spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add legend to show that red indicates the best model
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.8, label='Best Model'),
    Patch(facecolor='skyblue', alpha=0.8, label='Other Models')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Print results with best model highlighted
print("\nModel Accuracies:")
for i, (model, acc) in enumerate(zip(model_names, accuracies), 1):
    if acc == best_accuracy:
        print(f"{i}. {model}: {acc:.2f}% 🏆 BEST")
    else:
        print(f"{i}. {model}: {acc:.2f}%")

print(f"\n🎯 Best Performing Model: {best_model_name} with {best_accuracy:.2f}% accuracy")
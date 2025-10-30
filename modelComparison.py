import matplotlib.pyplot as plt
import numpy as np

# Before training data
before_training = [
    {"Model": "BERT Base", "Accuracy": 33.33},
    {"Model": "RoBERTa Base", "Accuracy": 30.00},
    {"Model": "BiomedBERT", "Accuracy": 26.67},
    {"Model": "PubMedBERT", "Accuracy": 13.33},
    {"Model": "Bio_ClinicalBERT", "Accuracy": 36.67},
    {"Model": "biobert-v1.1", "Accuracy": 30.00}
]

# After training data
after_training = [
    {"Model": "BERT Base", "Accuracy": 56.67},
    {"Model": "RoBERTa Base", "Accuracy": 53.33},
    {"Model": "BiomedBERT", "Accuracy": 46.67},
    {"Model": "PubMedBERT", "Accuracy": 50.0},
    {"Model": "Bio_ClinicalBERT", "Accuracy": 50.0},
    {"Model": "biobert-v1.1", "Accuracy": 40.0}
]

# Extract data for plotting
model_names = [item["Model"] for item in before_training]
before_accuracies = [item["Accuracy"] for item in before_training]
after_accuracies = [item["Accuracy"] for item in after_training]

# Find the best model after training
best_after_accuracy = max(after_accuracies)
best_after_index = after_accuracies.index(best_after_accuracy)
best_after_model = model_names[best_after_index]

# Set up the plot
plt.figure(figsize=(14, 8))

# Set the width of bars and positions
bar_width = 0.35
x_pos = np.arange(len(model_names))

# Create bars
before_bars = plt.bar(x_pos - bar_width/2, before_accuracies, bar_width,
                     label='Before Training', color='lightblue', alpha=0.7)
after_bars = plt.bar(x_pos + bar_width/2, after_accuracies, bar_width,
                    label='After Training', color='lightgreen', alpha=0.7)

# Highlight the best model after training
after_bars[best_after_index].set_color('red')
after_bars[best_after_index].set_alpha(0.9)
after_bars[best_after_index].set_edgecolor('darkred')
after_bars[best_after_index].set_linewidth(2)

# Add value labels on bars
for i, (before_bar, after_bar) in enumerate(zip(before_bars, after_bars)):
    # Before training values
    plt.text(before_bar.get_x() + before_bar.get_width()/2, before_bar.get_height() + 1,
            f'{before_accuracies[i]:.1f}%', ha='center', va='bottom', fontsize=9)

    # After training values
    plt.text(after_bar.get_x() + after_bar.get_width()/2, after_bar.get_height() + 1,
            f'{after_accuracies[i]:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold' if i == best_after_index else 'normal')

# Customize the plot
plt.title('Model Performance: Before vs After Training', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.xticks(x_pos, model_names, rotation=45, ha='right')
plt.ylim(0, max(after_accuracies) + 10)

# Add grid and legend
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)
plt.legend()

# Clean up spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add annotation for best model
plt.annotate(f'Best After Training: {best_after_model}\n{best_after_accuracy:.2f}%',
            xy=(best_after_index + bar_width/2, best_after_accuracy),
            xytext=(best_after_index, best_after_accuracy + 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Print detailed comparison
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON: BEFORE vs AFTER TRAINING")
print("="*70)
print(f"{'Model':<20} {'Before':<10} {'After':<10} {'Improvement':<12} {'Rank After'}")
print("-"*70)

# Sort models by after-training accuracy for ranking
sorted_indices = sorted(range(len(after_accuracies)), key=lambda i: after_accuracies[i], reverse=True)

for i, idx in enumerate(sorted_indices):
    model = model_names[idx]
    before = before_accuracies[idx]
    after = after_accuracies[idx]
    improvement = after - before

    if idx == best_after_index:
        rank = "🏆 1st (Best)"
    else:
        rank = f"{i+1}th"

    print(f"{model:<20} {before:<10.2f}% {after:<10.2f}% {improvement:+.2f}%{' ':>8} {rank}")

print("-"*70)

# Summary statistics
print(f"\n📊 SUMMARY:")
print(f"🏆 Best Model After Training: {best_after_model} ({best_after_accuracy:.2f}%)")
print(f"📈 Average Improvement: {np.mean([after_accuracies[i] - before_accuracies[i] for i in range(len(model_names))]):.2f}%")
print(f"🔥 Highest Improvement: {max([after_accuracies[i] - before_accuracies[i] for i in range(len(model_names))]):.2f}%")
print(f"📉 Lowest Improvement: {min([after_accuracies[i] - before_accuracies[i] for i in range(len(model_names))]):.2f}%")
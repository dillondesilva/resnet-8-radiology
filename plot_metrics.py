import pandas as pd
import matplotlib.pyplot as plt

# Update this file path to be any training metrics csv.
df = pd.read_csv('resnet-8-brain-tumor-classification-5-epoch/resnet-8-brain-tumor-classification-5-epoch-training-metrics.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
ax1.plot(df['Epochs'], df['Accuracy'], marker='o', linewidth=2, markersize=8)
ax1.set_title('Validation Accuracy', fontsize=12, pad=15)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim([0.7, 1.0])  # Set y-axis limits for better visualization
ax1.set_xticks([1, 2, 3, 4, 5])

# Plot loss
ax2.plot(df['Epochs'], df['Loss'], marker='o', color='red', linewidth=2, markersize=8)
ax2.set_title('Training Loss', fontsize=12, pad=15)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks([1, 2, 3, 4, 5])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('training_metrics_plot.png', dpi=300, bbox_inches='tight')
plt.close() 
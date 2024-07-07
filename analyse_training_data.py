import pandas as pd
import matplotlib.pyplot as plt

def merge_and_plot(csv_files):
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()
    max_epoch = 0  # Track the maximum epoch encountered so far
    
    # Iterate over the list of CSV files in the provided order and concatenate their contents
    for file in csv_files:
        df = pd.read_csv(file)
        # Adjust the epochs in the current file to continue from max_epoch
        df['Epochs'] += max_epoch
        max_epoch = df['Epochs'].max()  # Update max_epoch for the next file
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Plot the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Epochs'], merged_df['Loss'], marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.show()

# Example usage: provide the list of CSV file paths in the desired order
csv_files = [
    'brain-tumor-recognition/brain-tumor-recognition-training-metrics.csv', 
    '40-epoch-brain-tumor-recognition/40-epoch-brain-tumor-recognition-training-metrics.csv', 
    '50-epoch-brain-tumor-recognition-lr-change/50-epoch-brain-tumor-recognition-lr-change-training-metrics.csv'
]
merge_and_plot(csv_files)

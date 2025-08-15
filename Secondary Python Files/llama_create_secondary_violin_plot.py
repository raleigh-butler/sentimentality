import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import glob

def create_violin_plots(folder_path, column_name=None):
    """
    Create violin plots from all CSV files in the specified folder
    If column_name is specified, only plot that column
    """
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    # Create a figure with larger size
    plt.figure(figsize=(15, 8))
    
    # Read and combine all CSV files
    all_data = []
    for csv_file in csv_files:
        # Read CSV with specific column names
        df = pd.read_csv(csv_file, names=['Sentiment', 'Score'])
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Use Score column by default
        df = df[['Score']]
        
        # Map filenames to emotion labels
        emotion_map = {
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - LOVE': 'LOVE',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - SUBMISSION': 'SUBMISSION',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - AWE': 'AWE',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - DISAPPROVAL': 'DISAPPROVAL',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - REMORSE': 'REMORSE',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - CONTEMPT': 'CONTEMPT',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - AGGRESSIVENESS': 'AGGRESSIVENESS',
            'CHATGPT condensed output API CSV 1 & 2 all text individual rows - OPTIMISM': 'OPTIMISM'
        }
        
        # Find the matching key in emotion_map
        emotion_label = None
        for key, value in emotion_map.items():
            if value in file_name:
                emotion_label = value
                break
        
        # Add emotion label as identifier
        df['Source'] = emotion_label if emotion_label else file_name
        all_data.append(df)
    
    if not all_data:
        print("No valid data found to plot")
        return
        
    # Combine all data
    combined_df = pd.concat(all_data)
    
    # Create violin plot with custom colors
    plot_column = column_name if column_name else combined_df.columns[0]
    
    # Define color map for emotions
    emotion_colors = {
        'LOVE': '#f2f2b2',
        'SUBMISSION': '#dfebbb',
        'AWE': '#cfe5bf',
        'DISAPPROVAL': '#c3dfe0',
        'REMORSE': '#c3c7e4',
        'CONTEMPT': '#deb3d1',
        'AGGRESSIVENESS': '#f8c6bb',
        'OPTIMISM': '#fce9ad'
    }
    
    # Define the desired order
    desired_order = ['LOVE', 'SUBMISSION', 'AWE', 'DISAPPROVAL', 'REMORSE', 'CONTEMPT', 'AGGRESSIVENESS', 'OPTIMISM']
    
    # Create color palette in the desired order
    palette = [emotion_colors[emotion] for emotion in desired_order]
    
    # Reorder the data according to the desired order
    combined_df['Source'] = pd.Categorical(combined_df['Source'], categories=desired_order, ordered=True)
    
    # Create the violin plot with the ordered data
    sns.violinplot(x='Source', y=plot_column, hue='Source', data=combined_df, 
                  order=desired_order, palette=palette, legend=False, saturation=1)
    
    # Set axis labels and limits with larger font sizes
    plt.xlabel('')  # Remove Sentiment label
    plt.ylabel('Sentiment Score', fontsize=30, fontweight='bold')
    plt.ylim(-1, 1)
    
    # Set font sizes for axis labels and define y-axis ticks
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=24)
    plt.title('LLaMA 3.1 8B', fontsize=48, fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot to Desktop
    output_filename = os.path.expanduser('~/Desktop/secondary_violin_plot_comparison_llama.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"Created violin plot: {output_filename}")

def main():
    folder_path = os.path.expanduser("~/Desktop/SA_2_llama")
    create_violin_plots(folder_path)

if __name__ == "__main__":
    main()

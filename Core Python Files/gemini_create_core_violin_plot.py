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
            'Core Tones Data - ECSTASY': 'ECSTASY',
            'Core Tones Data - ADMIRATION': 'ADMIRATION',
            'Core Tones Data - TERROR': 'TERROR',
            'Core Tones Data - AMAZEMENT': 'AMAZEMENT',
            'Core Tones Data - GRIEF': 'GRIEF',
            'Core Tones Data - LOATHING': 'LOATHING',
            'Core Tones Data - RAGE': 'RAGE',
            'Core Tones Data - VIGILANCE': 'VIGILANCE'
        }
        
        # Find the matching key in emotion_map by checking if file_name contains any of the emotion names
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
        'ECSTASY': '#D5C402',
        'ADMIRATION': '#99AC00',
        'TERROR': '#5e8f00',
        'AMAZEMENT': '#327300',
        'GRIEF': '#005abf',
        'LOATHING': '#2d0031',
        'RAGE': '#690005',
        'VIGILANCE': '#9c2f01'
    }
    
    # Define the desired order
    desired_order = ['ECSTASY', 'ADMIRATION', 'TERROR', 'AMAZEMENT', 'GRIEF', 'LOATHING', 'RAGE', 'VIGILANCE']
    
    # Create color palette in the desired order
    palette = [emotion_colors[emotion] for emotion in desired_order]
    
    # Reorder the data according to the desired order
    combined_df['Source'] = pd.Categorical(combined_df['Source'], categories=desired_order, ordered=True)
    
    # Create the violin plot with the ordered data
    sns.violinplot(x='Source', y=plot_column, hue='Source', data=combined_df, 
                  order=desired_order, palette=palette, legend=False)
    
    # Set axis labels and limits with larger font sizes
    plt.xlabel('')  # Remove Sentiment label
    plt.ylabel('Sentiment Score', fontsize=30, fontweight='bold')
    plt.ylim(-1, 1)
    
    # Set font sizes for axis labels
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)  # Double size of y-axis numbers
    plt.title('Gemini 1.5 Flash', fontsize=48, fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot to Desktop
    output_filename = os.path.expanduser('~/Desktop/core_violin_plot_comparison_gemini.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"Created violin plot: {output_filename}")

def main():
    folder_path = os.path.expanduser("~/Desktop/SA_core_gemini")
    create_violin_plots(folder_path)

if __name__ == "__main__":
    main()

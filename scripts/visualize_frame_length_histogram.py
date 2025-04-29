import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_csv_log(csv_path, save_path='./plots', outlier_percentage=30):
    """
    Reads a CSV log file with frame statistics and generates visualizations.

    Args:
        csv_path (str): Path to the CSV file.
        save_path (str): Directory to save the plots.
        outlier_percentage (int): Percentage above the median frame length to detect outliers.
    """
    print(f"Reading data from: {csv_path}")
    os.makedirs(save_path, exist_ok=True)  # Create output dir if not exists

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    required_columns = ['Split', 'Sentence', 'Total Frames', 'Valid Frames', 'Zero Frames']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain the columns: {required_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    if df.empty:
        print("Error: CSV file is empty or contains no data rows.")
        return

    print(f"Successfully loaded {len(df)} rows.")

    # Convert columns to numeric
    for col in ['Total Frames', 'Valid Frames', 'Zero Frames']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # --- Frame Count per Video ---
    video_frame_counts = {}
    base_path = os.path.dirname(csv_path)  # Base path of the CSV file
    for split in ['train', 'dev', 'test']:
        split_df = df[df['Split'] == split]
        for video_folder in split_df['Sentence'].unique():

            # Build the path to the folder where the frames are located
            video_path = os.path.join(base_path, split, video_folder)

            # Check if the video_path is a directory and count .np files
            if os.path.isdir(video_path):
                print(f"Checking directory: {video_path}")  # Debug: print the directory path
                files_in_dir = [f for f in os.listdir(video_path) if f.endswith('.npy')]
                print(f"Found {len(files_in_dir)} .npy files in {video_path}")  # Debug: print the count of .np files
                if files_in_dir:
                    num_frames = len(files_in_dir)
                    video_frame_counts[video_folder] = num_frames
                else:
                    print(f"Warning: No .npy files found in {video_path}")
            else:
                print(f"Warning: '{video_path}' is not a valid directory.")

    if not video_frame_counts:
        print("Error: No valid frame counts found. Check the directory paths and file formats.")
        return

    # --- Outlier Detection ---
    all_frame_counts = list(video_frame_counts.values())
    if not all_frame_counts:
        print("Error: No frame counts to analyze for outliers.")
        return

    median = np.median(all_frame_counts)
    outlier_threshold = median + (median * outlier_percentage / 100)
    print(f"Outlier threshold set to {outlier_threshold:.2f} frames.")

    outliers = {video: frames for video, frames in video_frame_counts.items() if frames > outlier_threshold}

    # Save outliers to CSV with debug information
    if outliers:
        outlier_df = pd.DataFrame(outliers.items(), columns=['Video', 'Frames'])
        outlier_df.to_csv(os.path.join(save_path, 'outliers.csv'), index=False)
        print(f"Outliers saved to {os.path.join(save_path, 'outliers.csv')}")
    else:
        print("No outliers found to save.")

    # --- Plot 1: Stacked Bar Chart ---
    print("\nGenerating Stacked Bar Chart...")
    df_split_summary = df.groupby('Split')[['Total Frames', 'Valid Frames', 'Zero Frames']].sum()
    plt.figure(figsize=(10, 6))
    df_split_summary[['Valid Frames', 'Zero Frames']].plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'],
                                                           ax=plt.gca())
    plt.title('Frame Statistics per Data Split')
    plt.xlabel('Data Split')
    plt.ylabel('Number of Frames')
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'frames_per_split_stacked_bar.png'), dpi=300)
    print("Saved: frames_per_split_stacked_bar.png")

    # --- Plot 2: Histogram with Mean/Median ---
    print("Generating Histogram of Sequence Lengths with Mean/Median...")
    length_counts = df['Total Frames'].value_counts().sort_index()
    mean_val = df['Total Frames'].mean()
    median_val = df['Total Frames'].median()

    plt.figure(figsize=(12, 6))
    plt.bar(length_counts.index, length_counts.values, width=1.0, edgecolor='black', alpha=0.8)
    plt.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='purple', linestyle='-.', label=f'Median: {median_val:.1f}')
    plt.title('Histogram of Sequence Lengths (Frame Counts)')
    plt.xlabel('Frame Length per Sequence')
    plt.ylabel('Frequency (Number of Sequences)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'frame_length_histogram_with_stats.png'), dpi=300)
    print("Saved: frame_length_histogram_with_stats.png")

    # --- Plot 3: Split-wise Histograms ---
    print("Generating Split-wise Histograms...")
    for split in df['Split'].unique():
        split_df = df[df['Split'] == split]
        length_counts = split_df['Total Frames'].value_counts().sort_index()
        mean_val = split_df['Total Frames'].mean()
        median_val = split_df['Total Frames'].median()

        plt.figure(figsize=(12, 6))
        plt.bar(length_counts.index, length_counts.values, width=1.0, edgecolor='black', alpha=0.8)
        plt.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='purple', linestyle='-.', label=f'Median: {median_val:.1f}')
        plt.title(f'Frame Length Histogram for {split} Split')
        plt.xlabel('Frame Length per Sequence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'frame_length_histogram_{split}.png'), dpi=300)
        print(f"Saved: frame_length_histogram_{split}.png")

    # --- Plot 4: Valid Ratio per Split ---
    print("Generating Valid Frame Ratio Bar Chart...")
    df_split_summary['Valid Ratio'] = df_split_summary['Valid Frames'] / df_split_summary['Total Frames']
    df_split_summary['Valid Ratio'] = df_split_summary['Valid Ratio'].fillna(0)

    plt.figure(figsize=(8, 5))
    df_split_summary['Valid Ratio'].plot(kind='bar', color='teal', ax=plt.gca())
    plt.title('Valid Frame Ratio per Data Split')
    plt.xlabel('Data Split')
    plt.ylabel('Ratio (Valid / Total Frames)')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'valid_frame_ratio_per_split.png'), dpi=300)
    print("Saved: valid_frame_ratio_per_split.png")


# --- Example Run ---
csv_file_path = '/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_Benchmark_landmarks/summary.csv'
output_dir = '/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_Benchmark_landmarks/plots'
outlier_percentage = 60  # Adjust the percentage to detect outliers above median

visualize_csv_log(csv_file_path, save_path=output_dir, outlier_percentage=outlier_percentage)

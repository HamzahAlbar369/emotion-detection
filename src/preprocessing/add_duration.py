import pandas as pd
import librosa

def compute_duration(file_path, sr=16000):
    """
    Load an audio file and compute its duration (in seconds).
    """
    try:
        # Load the audio file at the specified sampling rate
        y, sr = librosa.load(file_path, sr=sr)
        # Compute and return the duration
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def append_duration_to_csv(csv_path, output_csv_path=None):
    """
    Load the processed features CSV, compute the duration for each file, and append a 'duration' column.
    Save the updated DataFrame to CSV.
    """
    # Load the existing processed CSV file
    df = pd.read_csv(csv_path)

    # Initialize a list to hold duration values
    durations = []

    # Iterate over each row and compute the duration
    for idx, row in df.iterrows():
        file_path = row.get('file_path')
        if not file_path:
            durations.append(None)
            continue
        duration = compute_duration(file_path)
        durations.append(duration)
    
    # Append the duration column to the DataFrame
    df['duration'] = durations

    # Decide whether to overwrite or create a new file (if output_csv_path is not provided, overwrite the original)
    if output_csv_path is None:
        output_csv_path = csv_path

    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

if __name__ == "__main__":
    # Path to your existing processed features CSV
    csv_path = "processed_crema_d_features.csv"
    # Optionally, specify an output path to avoid overwriting your original file
    output_csv_path = "processed_crema_d_features_with_duration.csv"
    
    append_duration_to_csv(csv_path, output_csv_path)

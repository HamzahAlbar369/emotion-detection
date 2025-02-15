import pandas as pd
from preprocess_audio import extract_features
import numpy as np
import os

# Load the metadata CSV
df = pd.read_csv('speech_dataset.csv')

# List to hold features and corresponding metadata
features_list = []
for idx, row in df.iterrows():
    file_path = row['file_path']
    try:
        feats = extract_features(file_path)
        features_list.append(feats)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        features_list.append(np.nan * np.ones( (len(feats), )))  # or skip

# Convert features to an array and add as new columns in the DataFrame (flattened features)
features_array = np.vstack(features_list)
# Create column names: For instance, mfcc_1 to mfcc_13, then zcr, pitch, energy, chroma_1 to chroma_12, spec_centroid, flux, hnr, f1, f2, f3.
col_names = (
    [f'mfcc_{i+1}' for i in range(13)] +
    ['zcr', 'pitch', 'energy'] +
    [f'chroma_{i+1}' for i in range(12)] +
    ['spec_centroid', 'flux', 'hnr', 'f1', 'f2', 'f3']
)
features_df = pd.DataFrame(features_array, columns=col_names)

# Merge features_df with the original metadata
df_final = pd.concat([df, features_df], axis=1)
df_final.to_csv('processed_crema_d_features.csv', index=False)
print("Processed features saved to 'processed_crema_d_features.csv'")

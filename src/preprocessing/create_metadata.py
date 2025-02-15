import os
import pandas as pd
import re

# Set the directory containing the CREMA-D audio files
crema_dir = 'data/crema-d/AudioWAV'

# Define a regex pattern for the filename (adjust if you need to catch anomalies)
# Expected format: <actor_id>_<sentence_id>_<emotion_id>_<emotion_intensity>.wav
pattern = re.compile(r'(?P<actor_id>\d{4})_(?P<sentence_id>[A-Z]{3})_(?P<emotion_id>[A-Z]{3})_(?P<intensity>[A-Z]{2,3})\.wav')

metadata = []
for file in os.listdir(crema_dir):
    if file.lower().endswith('.wav'):
        match = pattern.match(file)
        if match:
            meta = match.groupdict()
        else:
            # Handle anomalous filenames (e.g., missing underscore between sentence and emotion)
            # Attempt a simple fix or mark as error
            parts = file.replace('.wav','').split('_')
            if len(parts) == 3:
                # e.g., "1078_TIEDIS_XX" -> try splitting the second part into sentence and emotion
                actor_id = parts[0]
                # Assume first three letters are sentence, next three are emotion
                sentence_id = parts[1][:3]
                emotion_id = parts[1][3:6]
                intensity = parts[2]
                meta = {'actor_id': actor_id, 'sentence_id': sentence_id, 'emotion_id': emotion_id, 'intensity': intensity}
            else:
                # If still not matching, skip or log for manual review
                continue
        meta['file_path'] = os.path.join(crema_dir, file)
        metadata.append(meta)

df = pd.DataFrame(metadata)
df.to_csv('speech_dataset.csv', index=False)
print("Metadata CSV created with {} entries.".format(len(df)))

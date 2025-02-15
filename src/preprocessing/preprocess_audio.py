import os
import numpy as np
import librosa
import soundfile as sf
import parselmouth

def load_audio(file_path, sr=16000):
    """Load audio file and resample to sr."""
    y, _ = librosa.load(file_path, sr=sr)
    return y, sr

def trim_silence(y, top_db=60):
    """Trim leading and trailing silence using librosa."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def normalize_audio(y):
    """Normalize audio signal."""
    if np.max(np.abs(y)) == 0:
        return y
    return y / np.max(np.abs(y))

def export_audio(y, sr, output_path):
    """Export processed audio to WAV with 16-bit PCM."""
    sf.write(output_path, y, sr, subtype='PCM_16')

def extract_features(file_path):
    """Extract a comprehensive set of features from an audio file."""
    # Load and preprocess audio
    y, sr = load_audio(file_path)
    y = trim_silence(y)
    y = normalize_audio(y)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # 13-dimensional vector

    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Prosodic: Pitch estimation (using librosa.pyin)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = np.nanmean(f0[voiced_flag]) if np.any(voiced_flag) else 0

    # Prosodic: RMS Energy
    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Spectral Flux
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    flux_mean = np.mean(flux)

    # Harmonics-to-Noise Ratio (HNR) using parselmouth
    sound = parselmouth.Sound(file_path)
    try:
        hnr = np.mean(sound.to_harmonicity_cc().values)
    except Exception:
        hnr = 0

    # Formant Frequencies using parselmouth
    formant_object = sound.to_formant_burg()
    duration = sound.get_total_duration()
    times = np.linspace(0, duration, num=100)
    # Extract average of f1, f2, f3 over the time grid
    f1 = np.nanmean([formant_object.get_value_at_time(1, t) for t in times])
    f2 = np.nanmean([formant_object.get_value_at_time(2, t) for t in times])
    f3 = np.nanmean([formant_object.get_value_at_time(3, t) for t in times])

    # Compile all features into a single feature vector
    features = np.concatenate([
        mfccs_mean,
        [zcr_mean],
        [pitch_mean],
        [energy_mean],
        chroma_mean,
        [spec_centroid_mean],
        [flux_mean],
        [hnr],
        [f1, f2, f3]
    ])
    return features

if __name__ == "__main__":
    # Example usage: Process one file and print feature vector length.
    file_path = 'data/crema-d/AudioWAV/1078_TIE_FEA_XX.wav'
    features = extract_features(file_path)
    print("Extracted feature vector of length:", len(features))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

"""# Load an example audio file
y, sr = librosa.load(librosa.example('nutcracker'))

# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.tight_layout()
plt.show()

# Plot spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()"""

# Load audio
y, sr = librosa.load(librosa.example('nutcracker'))

# Volume (RMS)
rms = librosa.feature.rms(y=y)[0]
avg_rms = np.mean(rms)
print(f"Average RMS (volume): {avg_rms:.4f}")

# Pitch (using YIN)
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
avg_pitch = np.nanmean(f0)
print(f"Average pitch (Hz): {avg_pitch:.2f}")

# Spectral centroid (brightness)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
avg_centroid = np.mean(centroid)
print(f"Average spectral centroid: {avg_centroid:.2f} Hz")

# Jitter (pitch variation)
pitch_diff = np.abs(np.diff(f0))
jitter = np.nanmean(pitch_diff)
print(f"Jitter (mean absolute pitch difference): {jitter:.2f} Hz")


#folder extracter sample
import os
import librosa
import numpy as np

"""def extract_features_from_folders(parent_folder):
    data = {}
    for mood_folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, mood_folder)
        if not os.path.isdir(folder_path):
            continue
        data[mood_folder] = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.wav', '.mp3', '.ogg')):
                audio_path = os.path.join(folder_path, filename)
                y, sr = librosa.load(audio_path)
                rms = np.mean(librosa.feature.rms(y=y)[0])
                f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                avg_pitch = np.nanmean(f0)
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
                pitch_diff = np.abs(np.diff(f0))
                jitter = np.nanmean(pitch_diff)
                features = [rms, avg_pitch, centroid, jitter]
                data[mood_folder].append(features)
    return data

# Example usage:
parent_folder = r'd:\virtual_machine\ai_chat\my_audio_samples'  # Change to your parent folder path
features_dict = extract_features_from_folders(parent_folder)
print(features_dict)"""
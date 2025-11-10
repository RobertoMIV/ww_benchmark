import os
import subprocess

# Root folder containing the FLAC files
root_dir = "data/LibriSpeech/audio/train-100/"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".flac"):
            flac_path = os.path.join(dirpath, filename)
            wav_path = os.path.splitext(flac_path)[0] + ".wav"
            
            # Skip if already converted
            if os.path.exists(wav_path):
                continue

            # Run ffmpeg to convert
            subprocess.run([
                "ffmpeg", "-y", "-i", flac_path, wav_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Conversion complete!")
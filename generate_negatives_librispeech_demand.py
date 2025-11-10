import argparse
import json
import os
import random
import wave
import numpy as np
from tqdm import tqdm

def write_wav(path, audio, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_clipped = np.clip(audio, -1.0, 1.0)
    int16 = (audio_clipped * 32767.0).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        assert wf.getnchannels() == 1, "Expected mono"
        sr = wf.getframerate()
        nframes = wf.getnframes()
        audio = wf.readframes(nframes)
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)


def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-8:
        return clean

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    mixed = clean + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed


def choose_random_noise_segment(noise_files, length_samples, sr):
    noise_file = random.choice(noise_files)
    noise, noise_sr = read_wav(noise_file)
    if noise_sr != sr:
        raise ValueError(f"Noise SR mismatch: expected {sr}, got {noise_sr}")
    if len(noise) <= length_samples:
        reps = int(np.ceil(length_samples / len(noise)))
        noise = np.tile(noise, reps)
        return noise[:length_samples]
    start = random.randint(0, len(noise) - length_samples)
    return noise[start:start+length_samples]


def list_wavs(root):
    wavs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(dirpath, f))
    return sorted(wavs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-dir", default="data/LibriSpeech/audio/", help="Root of LibriSpeech wav files")
    parser.add_argument("--out-dir", default="data/negatives/librispeech")
    parser.add_argument("--manifest", default="manifests/negatives_librispeech.jsonl")
    parser.add_argument("--noise-dir", default="data/DEMAND/", help="DEMAND root directory")
    parser.add_argument("--snr-range", nargs=2, type=float, default=[5.0, 20.0])
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional cap on number of files to process")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    random.seed(123)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    noise_files = list_wavs(args.noise_dir) if args.noise_dir else []
    print(f"Found {len(noise_files)} noise wavs")

    all_libri = list_wavs(args.librispeech_dir)
    if args.max_files:
        all_libri = all_libri[:args.max_files]
    print(f"Processing {len(all_libri)} LibriSpeech files")

    with open(args.manifest, "w", encoding="utf-8") as mf:
        for i, src_path in tqdm(enumerate(all_libri), total=len(all_libri)):
            #print(src_path)
            audio, sr = read_wav(src_path)
            if sr != args.sample_rate:
                raise ValueError(f"Sample rate mismatch: expected {args.sample_rate}, got {sr}")

            basename = f"libri_{i:06d}.wav"
            out_path = os.path.join(args.out_dir, basename)

            noise_info = None
            if noise_files:
                snr_db = random.uniform(args.snr_range[0], args.snr_range[1])
                noise = choose_random_noise_segment(noise_files, len(audio), sr)
                audio_mix = mix_at_snr(audio, noise, snr_db)
                noise_info = {"snr_db": snr_db}
            else:
                audio_mix = audio

            write_wav(out_path, audio_mix, sr)

            duration_sec = len(audio_mix) / float(sr)
            entry = {
                "file": out_path,
                "sample_rate": sr,
                "keyword": "alexa",
                "events": [],  # no keyword events
                "source": "librispeech",
                "orig_path": src_path,
                "noise": noise_info,
                "duration_sec": duration_sec,
            }
            mf.write(json.dumps(entry) + "\n")

    print(f"Done. Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()

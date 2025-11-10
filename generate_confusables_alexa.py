import argparse
import json
import os
import random
import subprocess
import wave
import numpy as np

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
        assert wf.getnchannels() == 1
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


def synthesize_piper(piper_bin, model, text, out_wav, length_scale=1.0):
    cmd = [
        piper_bin,
        "--model", model,
        "--output_file", out_wav,
        "--length_scale", str(length_scale),
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"piper failed for {out_wav}:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--piper-bin", default="piper")
    parser.add_argument("--piper-models", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/confusables")
    parser.add_argument("--manifest", default="manifests/confusables_alexa.jsonl")
    parser.add_argument("--noise-dir", help="DEMAND root directory")
    parser.add_argument("--snr-range", nargs=2, type=float, default=[5.0, 20.0])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--utterances-per-phrase", type=int, default=20)
    args = parser.parse_args()

    random.seed(99)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    noise_files = list_wavs(args.noise_dir) if args.noise_dir else []

    # Hand-crafted list of phonetic neighbors of "Alexa"
    confusable_phrases = [
        "Alexis",
        "Alex",
        "Alexia",
        "Alexa's",
        "Electra",
        "Alyssa",
        "Elisa",
        "Alex, uh",
        "A Lexus",
        "Alex, ah",
        "Alyxa",     # will be read as some similar name usually
        "Alexa please",
        "Alexa play",
        "Alex and",
        "Alex at",
        "Alex uh"
    ]

    with open(args.manifest, "w", encoding="utf-8") as mf:
        utt_id = 0
        for model_path in args.piper_models:
            for phrase in confusable_phrases:
                for _ in range(args.utterances_per_phrase):
                    utt_id += 1
                    base = f"conf_{utt_id:05d}"
                    clean_path = os.path.join(args.out_dir, base + "_clean.wav")
                    final_path = os.path.join(args.out_dir, base + ".wav")

                    length_scale = random.uniform(0.9, 1.1)
                    synthesize_piper(args.piper_bin, model_path, phrase, clean_path, length_scale)

                    clean, sr = read_wav(clean_path)
                    if sr != args.sample_rate:
                        raise ValueError(f"Sample rate mismatch (expected {args.sample_rate}, got {sr})")

                    if noise_files:
                        snr_db = random.uniform(args.snr_range[0], args.snr_range[1])
                        noise_seg = choose_random_noise_segment(noise_files, len(clean), sr)
                        mixed = mix_at_snr(clean, noise_seg, snr_db)
                        noise_info = {"snr_db": snr_db}
                    else:
                        mixed = clean
                        noise_info = None

                    write_wav(final_path, mixed, sr)
                    os.remove(clean_path)

                    duration_sec = len(mixed) / float(sr)
                    entry = {
                        "file": final_path,
                        "sample_rate": sr,
                        "keyword": "alexa",
                        "events": [],  # important: no actual alexa event
                        "phrase": phrase,
                        "tag": "confusable",
                        "piper_model": model_path,
                        "length_scale": length_scale,
                        "noise": noise_info,
                        "duration_sec": duration_sec,
                    }
                    mf.write(json.dumps(entry) + "\n")

    print(f"Done. Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()

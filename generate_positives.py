import argparse
from glob import glob
import json
import os
import random
import subprocess
import wave
import numpy as np
import librosa
from tqdm import tqdm
# Helper to write int16 wav
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
        assert wf.getnchannels() == 1, "Expected mono audio"
        sr = wf.getframerate()
        # resample if needed
        if sr != 16000:
            audio, sr = librosa.load(path, sr=16000)
            return audio, sr
        nframes = wf.getnframes()
        audio = wf.readframes(nframes)
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)


def mix_at_snr(clean, noise, snr_db):
    # make noise at least as long as clean
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if noise_rms < 1e-8:
        return clean  # no noise effectively

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    mixed = clean + noise_scaled
    # normalize lightly if clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed


def choose_random_noise_segment(noise_files, length_samples, sr):
    # Pick a random file and random segment
    noise_file = random.choice(noise_files)
    noise, noise_sr = read_wav(noise_file)
    if noise_sr != sr:
        raise ValueError(f"Noise SR mismatch: expected {sr}, got {noise_sr}")
    if len(noise) <= length_samples:
        # just tile
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
    """
    Calls:
      piper --model model.onnx --output_file out.wav --length_scale <val> <<< "text"
    """
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        "piper",
        "--model", model,
        "--output_file", out_wav,
        "--length_scale", str(length_scale),
    ]
    # we’ll feed text on stdin
    proc = subprocess.run(
        cmd,
        input=(text+"\n").encode('utf-8')
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"piper failed for {out_wav}:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/positives_hey_ford", help="Output dir for positives")
    parser.add_argument("--manifest", default="manifests/positives.jsonl")
    parser.add_argument("--keyword", default="hey_ford", help="Keyword to synthesize")
    parser.add_argument("--num-utterances-per-voice", type=int, default=5)
    parser.add_argument("--noise-dir", default="data/DEMAND", help="DEMAND root directory for noise (optional)")
    parser.add_argument("--snr-range", nargs=2, type=float, default=[5.0, 20.0],
                        help="Min and max SNR in dB for mixing noise")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    outdir = args.out_dir
    manifest_out = args.manifest.replace(".jsonl", f"_{args.keyword}.jsonl")

    random.seed(42)
    os.makedirs(outdir if hasattr(args, "out_dir") else args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)

    noise_files = list_wavs(args.noise_dir) if args.noise_dir else []
    print(f"Found {len(noise_files)} noise wavs")

    manifest_f = open(manifest_out, "w", encoding="utf-8")

    # Text to synthesize
    keyword_text = args.keyword.replace("_", " ").title()

    utt_id = 0
    # Find Piper models
    models = glob("models/*.onnx")
    print(f"Generating positives using {len(models)} models")
    for model_path in models:
        for i in tqdm(range(args.num_utterances_per_voice), desc=f"Synthesizing for {os.path.basename(model_path)}"):
            utt_id += 1
            base_name = f"{args.keyword}_model{os.path.basename(model_path)}_{utt_id:04d}"
            clean_path = os.path.join(args.out_dir, base_name + "_clean.wav")
            final_path = os.path.join(args.out_dir, base_name + ".wav")

            # Slight variation in length_scale to change speaking rate
            length_scale = random.uniform(0.9, 1.1)

            # 1) synthesize clean
            synthesize_piper("piper", model_path, keyword_text, clean_path, length_scale=length_scale)

            # 2) load clean
            clean, sr = read_wav(clean_path)
            if sr != args.sample_rate:
                raise ValueError(f"Piper SR mismatch: expected {args.sample_rate}, got {sr}")

            # 3) mix noise if available
            if noise_files:
                snr_db = random.uniform(args.snr_range[0], args.snr_range[1])
                noise_seg = choose_random_noise_segment(noise_files, len(clean), sr)
                mixed = mix_at_snr(clean, noise_seg, snr_db)
                write_wav(final_path, mixed, sr)
                os.remove(clean_path)
                noise_info = {"snr_db": snr_db}
            else:
                write_wav(final_path, clean, sr)
                noise_info = None

            duration_sec = len(clean) / float(sr)

            # event: whole file is the keyword
            event = {
                "file": final_path,
                "sample_rate": sr,
                "keyword": args.keyword,
                "events": [
                    {
                        "start_sec": 0.0,
                        "end_sec": duration_sec,
                        "tag": "keyword"
                    }
                ],
                "noise": noise_info,
                "piper_model": model_path,
                "length_scale": length_scale,
            }
            manifest_f.write(json.dumps(event) + "\n")

    manifest_f.close()
    print(f"Done. Wrote manifest to {manifest_out}")


if __name__ == "__main__":
    main()

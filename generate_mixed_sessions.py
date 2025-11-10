import argparse, json, os, random, wave, numpy as np

def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr

def write_wav(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = np.clip(audio, -1, 1)
    int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(int16.tobytes())

def load_manifest(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives-manifest", default="manifests/positives_alexa.jsonl")
    parser.add_argument("--negatives-manifest", default="manifests/negatives_librispeech.jsonl")
    parser.add_argument("--out-dir", default="data/sessions")
    parser.add_argument("--manifest-out", default="manifests/sessions_labels.jsonl")
    parser.add_argument("--target-length-min", type=float, default=10.0)
    parser.add_argument("--num-sessions", type=int, default=3)
    parser.add_argument("--num-positives-per-session", type=int, default=3)
    args = parser.parse_args()

    sr = 16000
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)

    positives = load_manifest(args.positives_manifest if hasattr(args, "positives_manifest") else args.positives_manifest)
    negatives = load_manifest(args.negatives_manifest if hasattr(args, "negatives_manifest") else args.negatives_manifest)

    with open(args.manifest_out, "w") as mf:
        for sid in range(args.num_sessions):
            session_audio = []
            session_events = []
            total_samples = 0
            target_samples = int(args.target_length_min * 60 * sr)
            # fill with negative chunks
            while total_samples < target_samples:
                neg = random.choice(negatives)
                a, s = read_wav(neg["file"])
                if s != sr:
                    continue
                session_audio.append(a)
                total_samples += len(a)
            concat_audio = np.concatenate(session_audio)[:target_samples]

            # insert positives at random times
            num_pos = args.num_positives_per_session
            for _ in range(num_pos):
                pos = random.choice(positives)
                pa, s = read_wav(pos["file"])
                dur = len(pa) / sr
                insert_start = random.uniform(2.0, (target_samples / sr) - dur - 2.0)
                insert_idx = int(insert_start * sr)
                insert_end = insert_idx + len(pa)
                if insert_end > len(concat_audio):
                    continue
                concat_audio[insert_idx:insert_end] += pa
                # lightly renormalize if clipping
                peak = np.max(np.abs(concat_audio))
                if peak > 1.0:
                    concat_audio = concat_audio / peak
                session_events.append({
                    "start_sec": insert_start,
                    "end_sec": insert_start + dur,
                    "tag": "alexa"
                })

            out_path = os.path.join(args.out_dir, f"session_{sid:02d}.wav")
            write_wav(out_path, concat_audio, sr)
            entry = {
                "file": out_path,
                "sample_rate": sr,
                "events": session_events,
                "duration_sec": len(concat_audio)/sr
            }
            mf.write(json.dumps(entry) + "\n")

    print(f"Generated {args.num_sessions} mixed sessions with positives inserted.")

if __name__ == "__main__":
    main()

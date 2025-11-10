import argparse, json, os, time, psutil, threading
import numpy as np
from statistics import mean
from collections import defaultdict
from typing import List, Dict, Tuple

from tqdm import tqdm
from engines import Engine, Engines  # <- your engines module
from dataclasses import dataclass

@dataclass
class Trigger:
    time_sec: float
    confidence: float = 1.0

# --- Helper: read wav ---
import wave
def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data, sr


# --- Resource monitor thread ---
class ResourceMonitor(threading.Thread):
    def __init__(self, pid=None, interval=1.0):
        super().__init__()
        self.daemon = True
        self.process = psutil.Process(pid or os.getpid())
        self.interval = interval
        self.cpu = []
        self.mem = []
        self.running = True

    def run(self):
        self.process.cpu_percent(interval=None)  # reset baseline
        while self.running:
            self.cpu.append(self.process.cpu_percent(interval=None))
            self.mem.append(self.process.memory_info().rss / 1024**2)  # MB
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def summary(self):
        return {
            "cpu_avg_percent": mean(self.cpu) if self.cpu else 0,
            "mem_avg_mb": mean(self.mem) if self.mem else 0
        }


# --- Match triggers to ground truth events ---
def match_triggers_to_events(triggers: List[Trigger], events: List[Dict], tolerance=1.0):
    """Match triggers to events allowing ±tolerance seconds."""
    hits = 0
    misses = 0
    latency_sum = 0.0
    matched = set()

    for e in events:
        t_start = e["start_sec"]
        t_end = e["end_sec"]
        window_start = t_start - tolerance
        window_end = t_end + tolerance
        found = False
        for tr in triggers:
            if window_start <= tr.time_sec <= window_end:
                found = True
                hits += 1
                latency_sum += max(0.0, tr.time_sec - t_start)
                matched.add(tr.time_sec)
                break
        if not found:
            misses += 1

    false_triggers = [t for t in triggers if t.time_sec not in matched]
    return hits, misses, latency_sum, false_triggers


def evaluate_engine(engine, manifest, sr, hop_len, tolerance=1.0):
    proc = psutil.Process(os.getpid())
    # Optional: enforce single core
    try:
        proc.cpu_affinity([0])
    except Exception:
        pass

    monitor = ResourceMonitor()
    monitor.start()

    total_events, total_hits, total_misses, total_latency, total_false, total_dur = 0, 0, 0, 0, 0, 0
    start_all = time.perf_counter()
    triggers_per_file = {}

    for entry in manifest:
        print(f"Running on {entry['file']} ...")
        triggers = process_audio(engine, entry["file"], sr, hop_len)
        hits, misses, latency_sum, false_trigs = match_triggers_to_events(triggers, entry["events"], tolerance)
        total_events += len(entry["events"])
        total_hits += hits
        total_misses += misses
        total_latency += latency_sum
        total_false += len(false_trigs)
        total_dur += entry["duration_sec"]
        triggers_per_file[entry["file"]] = triggers

    total_time = time.perf_counter() - start_all
    monitor.stop()
    time.sleep(0.2)  # allow monitor to flush
    res = monitor.summary()

    rtf = total_time / (total_dur if total_dur > 0 else 1)

    metrics = {
        "total_events": total_events,
        "hits": total_hits,
        "misses": total_misses,
        "miss_rate": total_misses / total_events if total_events else 0,
        "false_alarms": total_false,
        "false_alarm_rate_per_hour": total_false / (total_dur / 3600.0) if total_dur else 0,
        "avg_latency_sec": total_latency / total_hits if total_hits else None,
        "cpu_avg_percent": res["cpu_avg_percent"],
        "mem_avg_mb": res["mem_avg_mb"],
        "real_time_factor": rtf,
        "total_runtime_sec": total_time,
    }
    return metrics, triggers_per_file


def process_audio(engine: Engine, wav_path: str, sr: int, hop_length: int):
    """Stream through audio, call engine.process(), collect trigger times."""
    audio, s = read_wav(wav_path)
    assert s == sr, f"Expected {sr}, got {s}"
    frame_len = engine.frame_length(type(engine))
    hop_len = hop_length
    triggers = []

    # Slide through audio
    num_samples = len(audio)
    i = 0
    pbar = tqdm(total=num_samples)
    #while i + frame_len <= num_samples:
    for i in range(0, num_samples - frame_len + 1, hop_len):
        frame = audio[i:i + frame_len]
        if engine.process(frame):
            triggers.append(Trigger(i / sr))
        pbar.update(hop_len)
    pbar.close()

    return triggers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-manifest", default="manifests/sessions_labels.jsonl", help="Path to sessions_labels.jsonl")
    parser.add_argument("--keyword", default="alexa")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--output-json", default="benchmark_summary.json")
    args = parser.parse_args()
    # Restrict backend libraries to 1 thread for real single-core testing
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = "1"

    # Initialize engine
    
    for engine_type in Engines:
        print(f"\n=== Evaluating with engine: {engine_type.name} ===")
        if engine_type is Engines.PORCUPINE:
            engine = Engine.create(engine_type, args.keyword, args.sensitivity)
        else:
            engine = Engine.create(engine_type, args.keyword, args.sensitivity)



        manifest = [json.loads(line) for line in open(args.sessions-manifest if hasattr(args,"sessions-manifest") else args.sessions_manifest)]

        print(f"Evaluating {len(manifest)} files...")
        metrics, triggers_per_file = evaluate_engine(engine, manifest, args.sample_rate, args.hop_length, args.tolerance)
        print("\n=== Benchmark Results ===")
        for k,v in metrics.items():
            print(f"{k:35s}: {v}")

        out_log = os.path.splitext(args.sessions_manifest)[0] + f"_{engine_type.name}_triggers.jsonl"
        with open(out_log, "w") as f:
            for file, triggers in triggers_per_file.items():
                f.write(json.dumps({"file": file, "triggers": [t.__dict__ for t in triggers]}) + "\n")
        out_log_resources = os.path.splitext(args.sessions_manifest)[0] + f"_{engine_type.name}_resources.jsonl"
        
        with open(out_log_resources, "w") as f:
            f.write(json.dumps(metrics, indent=2))
        engine.release()

        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
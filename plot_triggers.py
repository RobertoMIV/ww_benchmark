import json, argparse, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def classify_triggers(events, triggers, tolerance=1.0):
    """Classify triggers as TP (true positive), FP (false alarm), FN (miss)."""
    tp, fp = [], []
    matched_events = set()

    # flatten triggers to seconds
    trigger_times = [t["time_sec"] for t in triggers]

    for tr in trigger_times:
        matched = False
        for idx, ev in enumerate(events):
            if ev["start_sec"] - tolerance <= tr <= ev["end_sec"] + tolerance:
                matched = True
                matched_events.add(idx)
                tp.append(tr)
                break
        if not matched:
            fp.append(tr)

    fn = [ev for i, ev in enumerate(events) if i not in matched_events]
    return tp, fp, fn


def plot_session(gt_entry, triggers, out_path, tolerance=1.0, model_name="EFFICIENTWORD"):
    events = gt_entry["events"]
    tp, fp, fn = classify_triggers(events, triggers, tolerance)
    duration = gt_entry.get("duration_sec", max(ev["end_sec"] for ev in events) + 1)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_title(os.path.basename(gt_entry["file"]) + f" - {model_name}")
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])

    # ground truth intervals
    for ev in events:
        ax.add_patch(
            mpatches.Rectangle(
                (ev["start_sec"], 0.2),
                ev["end_sec"] - ev["start_sec"],
                0.6,
                color="lime",
                alpha=0.5,
            )
        )

    # triggers
    for t in tp:
        ax.axvline(t, color="green", linestyle="--", ymax=0.9, label="Correct Trigger" if "Correct Trigger" not in [l.get_label() for l in ax.lines] else "")
    for t in fp:
        ax.axvline(t, color="red", linestyle=":", ymax=0.9, label="False Alarm" if "False Alarm" not in [l.get_label() for l in ax.lines] else "")

    # missed events (FN)
    for ev in fn:
        x = (ev["start_sec"] + ev["end_sec"]) / 2
        ax.plot(x, 0.5, marker="x", color="black", markersize=10, mew=2)

    ax.set_xlim(0, duration)

    legend = [
        mpatches.Patch(color="lime", alpha=0.5, label="Ground Truth"),
        plt.Line2D([], [], color="green", linestyle="--", label="Correct Trigger"),
        plt.Line2D([], [], color="red", linestyle=":", label="False Alarm"),
        plt.Line2D([], [], color="black", marker="x", linestyle="", label="Missed Event"),
    ]
    ax.legend(handles=legend, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="manifests/sessions_labels.jsonl", help="sessions_labels.jsonl")
    #parser.add_argument("--predictions", default="manifests/sessions_labels_EFFICIENTWORD_triggers.jsonl", help="*_triggers.jsonl")
    parser.add_argument("--model-name", default="EFFICIENTWORD")
    parser.add_argument("--out-dir", default="plots_timeline")
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt_data = load_jsonl(args.ground_truth)
    filename = args.model_name
    print(f"Loading predictions from {filename} ...")
    pred_data = {p["file"]: p["triggers"] for p in load_jsonl(f"manifests/sessions_labels_{filename}_triggers.jsonl")}

    for entry in gt_data:
        wav_path = entry["file"]
        triggers = pred_data.get(wav_path, [])
        filename_png = os.path.basename(wav_path).split(".wav")[0] + f"_{filename}_timeline.png"
        out_path = os.path.join(
            args.out_dir, filename_png
        )
        plot_session(entry, triggers, out_path, tolerance=args.tolerance,model_name=args.model_name)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
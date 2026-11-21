"""
Run Qwen3-ASR-1.7B on your own audio files and measure WER, CER, time and memory.

SETUP (Google Colab with a GPU, or your own machine):
    pip install -U qwen-asr soundfile
    (Colab: Runtime > Change runtime type > T4 GPU)

YOUR DATA (put in one folder, default name: audio/):
    audio/s1.wav   <- the recording (wav/mp3/flac)
    audio/s1.txt   <- the correct transcript (ground truth) of that recording, plain text
    audio/s2.wav
    audio/s2.txt
    ...

RUN:
    python run_qwen3_asr_experiment.py --audio_dir audio

OUTPUT (all numbers come from your real run):
    results.csv, results.json   -> every measurement + the model's text
    results_table.tex           -> table rows for the LaTeX paper
    summary.tex                 -> macros (average WER, GPU name, ...) for the LaTeX paper
"""
import argparse
import csv
import glob
import json
import os
import re
import time

AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


# ---------- text normalisation and metrics (no extra library needed) ----------
def normalize(text: str) -> str:
    """Lowercase, remove punctuation (keep apostrophes), collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (substitution, deletion, insertion = 1)."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def clean_reference(text: str) -> str:
    """Keep only the spoken words of a reference file.
    Removes subtitle timestamps ("00:00:00,880 --> 00:00:05,160"), tags like [Speaker 0] or [rustling],
    and filler sounds (uh, um, hmm, Arabic aaa). Describe this cleaning in the paper."""
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\b(?:uh+|um+|uhm+|erm?|hmm+|mm+)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)آ+(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


# ---------- helpers ----------
def audio_duration(path: str) -> float:
    """Length of the audio in seconds (works for wav, mp3, flac and m4a)."""
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        pass
    try:
        import librosa  # installed together with qwen-asr
        return float(librosa.get_duration(path=path))
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())


def tex_escape(s: str) -> str:
    return s.replace("\\", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_latex(rows, meta, out_dir="."):
    """Write results_table.tex and summary.tex from the measured rows."""
    lines = []
    for r in rows:
        lines.append(
            f"{tex_escape(r['id'])} & {r['duration_s']:.1f} & {r['time_s']:.2f} & "
            f"{r['rtf']:.3f} & {100 * r['wer']:.1f} & {100 * r['cer']:.1f}"
        )
    with open(os.path.join(out_dir, "results_table.tex"), "w", encoding="utf8") as f:
        # last row has no "\\\\": the LaTeX template adds it after \\input
        f.write(" \\\\\n".join(lines))

    n = len(rows)
    tot_dur = sum(r["duration_s"] for r in rows)
    tot_time = sum(r["time_s"] for r in rows)
    avg_wer = 100 * sum(r["wer"] for r in rows) / n
    avg_cer = 100 * sum(r["cer"] for r in rows) / n
    macros = {
        "nfiles": str(n),
        "totaldur": f"{tot_dur:.1f}",
        "avgwer": f"{avg_wer:.1f}",
        "avgcer": f"{avg_cer:.1f}",
        "avgrtf": f"{tot_time / tot_dur:.3f}",
        "loadtime": f"{meta['load_time_s']:.1f}",
        "peakmem": f"{meta['peak_gpu_mem_gb']:.1f}" if meta["peak_gpu_mem_gb"] is not None else "n/a",
        "devicename": tex_escape(meta["device_name"]),
        "torchver": tex_escape(meta["torch_version"]),
        "dtypename": tex_escape(meta["dtype"]),
    }
    with open(os.path.join(out_dir, "summary.tex"), "w", encoding="utf8") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return macros


# ---------- main experiment ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--language", default=None, help='e.g. "English"; default None = automatic detection')
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="increase for long audio")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="on a Colab T4 GPU, try float16 if bfloat16 is slow or fails")
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = getattr(torch, args.dtype) if use_cuda else torch.float32
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    files = sorted(p for p in glob.glob(os.path.join(args.audio_dir, "*")) if p.lower().endswith(AUDIO_EXT))
    if not files:
        raise SystemExit(f"No audio files found in {args.audio_dir}/")

    print(f"Loading {args.model} on {device_name} ({dtype}) ...")
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f} s")

    # warm-up run (not counted) so the first file is not slower because of GPU start-up
    model.transcribe(audio=files[0], language=args.language)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for path in files:
        base = os.path.splitext(path)[0]
        ref_path = base + ".txt"
        if not os.path.exists(ref_path):
            print(f"[skip] no ground-truth file for {path}")
            continue
        ref_raw = open(ref_path, encoding="utf8").read().strip()
        ref = clean_reference(ref_raw)
        dur = audio_duration(path)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        res = model.transcribe(audio=path, language=args.language)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - t1

        hyp = res[0].text
        row = {
            "id": os.path.splitext(os.path.basename(path))[0],
            "duration_s": dur,
            "time_s": elapsed,
            "rtf": elapsed / dur,
            "wer": wer(ref, hyp),
            "cer": cer(ref, hyp),
            "language": str(res[0].language),
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"{row['id']:>15} | {dur:6.1f}s | {elapsed:6.2f}s | RTF {row['rtf']:.3f} | "
              f"WER {100 * row['wer']:5.1f}% | CER {100 * row['cer']:5.1f}% | {row['language']}")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if use_cuda else None
    meta = {
        "model": args.model,
        "device_name": device_name,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "load_time_s": load_time,
        "peak_gpu_mem_gb": peak,
    }

    with open("results.json", "w", encoding="utf8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)
    with open("results.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    macros = write_latex(rows, meta)

    print("\n=== SUMMARY (copy into your paper) ===")
    print(f"files: {macros['nfiles']}, total audio: {macros['totaldur']} s")
    print(f"average WER: {macros['avgwer']}%  | average CER: {macros['avgcer']}%")
    print(f"overall RTF: {macros['avgrtf']}  | load time: {macros['loadtime']} s | peak GPU memory: {macros['peakmem']} GB")
    print("Wrote results.csv, results.json, results_table.tex, summary.tex")


if __name__ == "__main__":
    main()

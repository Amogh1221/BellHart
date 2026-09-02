#!/usr/bin/env python3
"""
plot.py  —  BellHart Training Log Visualizer & Metric Dashboard
===============================================================
Parses training_log.txt / logs.txt to generate comprehensive, publication-ready
visual analytics of pre-training metrics:
  1. Training & Validation Loss Curve (with EMA / rolling trendlines)
  2. Validation Perplexity (PPL)
  3. Learning Rate Warmup & Decay Schedule
  4. Gradient Norm Dynamics & Stability
  5. Throughput Performance (Tokens / Second)
  6. Cumulative Tokens Processed

Usage:
  python plot.py
  python plot.py --log_file logs.txt --out training_metrics.png
  python plot.py --log_file logs/training_log.txt --smooth 20
"""

import os
import sys
import re
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────────────────────────────────────
# Visual Theme Configuration (Modern Dark Palette)
# ──────────────────────────────────────────────────────────────────────────────

THEME = {
    "bg": "#0d1117",            # Canvas background
    "card_bg": "#161b22",       # Subplot axis background
    "grid": "#30363d",          # Subplot gridlines
    "text": "#c9d1d9",          # Normal labels and ticks
    "title": "#58a6ff",         # Titles and highlights
    "accent_blue": "#38bdf8",   # Primary train loss / metrics
    "accent_cyan": "#22d3ee",   # Learning rate
    "accent_green": "#4ade80",  # Throughput
    "accent_orange": "#fb923c", # Validation loss
    "accent_pink": "#f472b6",   # Perplexity
    "accent_purple": "#a78bfa", # Gradient norm
    "accent_gold": "#fbbf24",   # Best markers
}


def apply_dark_theme():
    """Applies modern dark theme aesthetics to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["card_bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["text"],
        "axes.grid": True,
        "grid.color": THEME["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
        "text.color": THEME["text"],
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.facecolor": THEME["card_bg"],
        "legend.edgecolor": THEME["grid"],
        "legend.fontsize": 9,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Log Parser Engine
# ──────────────────────────────────────────────────────────────────────────────

def parse_log_file(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Parses both step-by-step progress lines and multi-line evaluation blocks.
    Automatically handles restarts by deduplicating earlier overlapping steps.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    # Regex for step lines:
    # [2026-08-28 03:59:53] STEP       1/150000 | Tokens: 163,840 | loss=23.4341 | lr=3.33e-07 | grad_norm=2.301 | tok/s=583
    step_pattern = re.compile(
        r"\[(?P<time>[\d\- :]+)\]\s+STEP\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+"
        r"Tokens:\s+(?P<tokens>[\d,]+)\s+\|\s+"
        r"loss=(?P<loss>[\d\.]+)\s+\|\s+"
        r"lr=(?P<lr>[\d\.eE\+\-]+)\s+\|\s+"
        r"grad_norm=(?P<grad_norm>[\d\.]+)\s+\|\s+"
        r"tok/s=(?P<tok_sec>[\d,]+)"
    )

    # Regex patterns for evaluation blocks:
    eval_header = re.compile(r"EVALUATION @ Step (?P<step>\d+)\s+/\s+(?P<total>\d+)")
    eval_tokens = re.compile(r"Tokens Processed\s+:\s+(?P<tokens>[\d,]+)")
    eval_train_loss = re.compile(r"Train Loss\s+:\s+(?P<loss>[\d\.]+)")
    eval_val_loss = re.compile(r"Val Loss\s+:\s+(?P<loss>[\d\.]+)")
    eval_ppl = re.compile(r"Perplexity\s+:\s+(?P<ppl>[\d\.]+)")

    step_data = {}  # step -> dict (ensures latest restart overwrites old step data)
    eval_data = {}  # step -> dict

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 1. Parse Step Lines
    for match in step_pattern.finditer(content):
        step = int(match.group("step"))
        tokens = int(match.group("tokens").replace(",", ""))
        loss = float(match.group("loss"))
        lr = float(match.group("lr"))
        grad_norm = float(match.group("grad_norm"))
        tok_sec = float(match.group("tok_sec").replace(",", ""))
        ts_str = match.group("time")

        step_data[step] = {
            "step": step,
            "tokens": tokens,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tok_sec": tok_sec,
            "timestamp": ts_str,
        }

    # 2. Parse Evaluation Blocks
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = eval_header.search(line)
        if match:
            step = int(match.group("step"))
            entry = {"step": step}
            # Scan following lines within evaluation block
            for j in range(i + 1, min(i + 20, len(lines))):
                sub = lines[j]
                if "Tokens Processed" in sub:
                    t_match = eval_tokens.search(sub)
                    if t_match:
                        entry["tokens"] = int(t_match.group("tokens").replace(",", ""))
                elif "Train Loss" in sub:
                    tl_match = eval_train_loss.search(sub)
                    if tl_match:
                        entry["train_loss"] = float(tl_match.group("loss"))
                elif "Val Loss" in sub:
                    vl_match = eval_val_loss.search(sub)
                    if vl_match:
                        entry["val_loss"] = float(vl_match.group("loss"))
                elif "Perplexity" in sub:
                    p_match = eval_ppl.search(sub)
                    if p_match:
                        entry["ppl"] = float(p_match.group("ppl"))
                elif "════════" in sub and j > i + 1:
                    break
            eval_data[step] = entry
        i += 1

    # Convert sorted dictionaries to numpy arrays
    sorted_steps = sorted(step_data.keys())
    if not sorted_steps:
        raise ValueError(f"No step training data could be parsed from {filepath}.")

    steps_dict = {
        "step": np.array([step_data[s]["step"] for s in sorted_steps]),
        "tokens": np.array([step_data[s]["tokens"] for s in sorted_steps]),
        "loss": np.array([step_data[s]["loss"] for s in sorted_steps]),
        "lr": np.array([step_data[s]["lr"] for s in sorted_steps]),
        "grad_norm": np.array([step_data[s]["grad_norm"] for s in sorted_steps]),
        "tok_sec": np.array([step_data[s]["tok_sec"] for s in sorted_steps]),
    }

    sorted_evals = sorted(eval_data.keys())
    evals_dict = {
        "step": np.array([eval_data[s]["step"] for s in sorted_evals if "val_loss" in eval_data[s]]),
        "tokens": np.array([eval_data[s].get("tokens", 0) for s in sorted_evals if "val_loss" in eval_data[s]]),
        "train_loss": np.array([eval_data[s].get("train_loss", np.nan) for s in sorted_evals if "val_loss" in eval_data[s]]),
        "val_loss": np.array([eval_data[s]["val_loss"] for s in sorted_evals if "val_loss" in eval_data[s]]),
        "ppl": np.array([eval_data[s].get("ppl", np.exp(eval_data[s]["val_loss"])) for s in sorted_evals if "val_loss" in eval_data[s]]),
    }

    return steps_dict, evals_dict


def moving_average(data: np.ndarray, window_size: int = 25) -> np.ndarray:
    """Computes a smooth moving average with edge padding."""
    if len(data) < window_size:
        return data
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode="valid")
    # Pad beginning so lengths match
    pad = np.full(window_size - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def create_dashboard(
    steps: Dict[str, np.ndarray],
    evals: Dict[str, np.ndarray],
    output_path: str = "training_plots.png",
    smooth_window: int = 25,
    show: bool = False,
):
    """Renders the 6-panel BellHart training dashboard and saves high-res image."""
    apply_dark_theme()

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.18, left=0.06, right=0.96, top=0.92, bottom=0.06)

    step_nums = steps["step"]
    current_step = step_nums[-1]
    total_tokens = steps["tokens"][-1]
    best_loss = np.min(steps["loss"])
    best_val = np.min(evals["val_loss"]) if len(evals["val_loss"]) > 0 else np.nan
    best_ppl = np.min(evals["ppl"]) if len(evals["ppl"]) > 0 else np.nan

    # Figure Header
    fig.suptitle(
        f"BellHart Pre-Training Metrics  •  Step {current_step:,}  •  {total_tokens / 1e6:,.1f}M Tokens  •  "
        f"Min Train Loss: {best_loss:.4f}  •  Best Val PPL: {best_ppl:.2f}",
        fontsize=15,
        fontweight="bold",
        color=THEME["title"],
    )

    # ── Panel 1: Cross-Entropy Loss Curve ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    smoothed_loss = moving_average(steps["loss"], window_size=smooth_window)
    ax1.plot(step_nums, steps["loss"], color=THEME["accent_blue"], alpha=0.25, label="Step Loss (Raw)", lw=1.0)
    ax1.plot(step_nums, smoothed_loss, color=THEME["accent_blue"], lw=2.0, label=f"Train Loss (MA-{smooth_window})")

    if len(evals["val_loss"]) > 0:
        ax1.plot(evals["step"], evals["val_loss"], color=THEME["accent_orange"], marker="o", lw=2.0, ms=6, label="Val Loss (Eval)")
        best_idx = np.argmin(evals["val_loss"])
        ax1.scatter(
            [evals["step"][best_idx]],
            [evals["val_loss"][best_idx]],
            color=THEME["accent_gold"],
            s=120,
            zorder=5,
            edgecolors=THEME["bg"],
            label=f"Best Val: {evals['val_loss'][best_idx]:.4f}",
        )

    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Optimizer Steps")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # ── Panel 2: Validation Perplexity (PPL) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if len(evals["ppl"]) > 0:
        ax2.plot(evals["step"], evals["ppl"], color=THEME["accent_pink"], marker="s", lw=2.0, ms=6, label="Validation PPL")
        ax2.set_title("Validation Perplexity (Log Scale)", fontweight="bold")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.set_xlabel("Optimizer Steps")
        ax2.set_ylabel("Perplexity (exp(Loss))")
        ax2.legend(loc="upper right")
    else:
        # Fallback to token-level loss if no evals available yet
        ax2.plot(step_nums, np.exp(np.clip(smoothed_loss, 0, 10)), color=THEME["accent_pink"], lw=1.8)
        ax2.set_title("Estimated Training Perplexity", fontweight="bold")
        ax2.set_yscale("log")
        ax2.set_xlabel("Optimizer Steps")
        ax2.set_ylabel("Perplexity")
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # ── Panel 3: Learning Rate Schedule ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(step_nums, steps["lr"], color=THEME["accent_cyan"], lw=2.0, label="Learning Rate")
    ax3.set_title("Learning Rate Schedule", fontweight="bold")
    ax3.set_xlabel("Optimizer Steps")
    ax3.set_ylabel("Learning Rate")
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax3.legend(loc="lower right")

    # ── Panel 4: Gradient Norm Dynamics ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    smoothed_gn = moving_average(steps["grad_norm"], window_size=smooth_window)
    ax4.plot(step_nums, steps["grad_norm"], color=THEME["accent_purple"], alpha=0.25, lw=1.0, label="Grad Norm (Raw)")
    ax4.plot(step_nums, smoothed_gn, color=THEME["accent_purple"], lw=2.0, label=f"Grad Norm (MA-{smooth_window})")
    ax4.axhline(1.0, color="#f87171", linestyle=":", alpha=0.7, label="Clip Threshold (1.0)")
    ax4.set_title("Gradient Norm Stability", fontweight="bold")
    ax4.set_xlabel("Optimizer Steps")
    ax4.set_ylabel("L2 Norm")
    ax4.set_ylim(0, max(2.5, min(np.percentile(steps["grad_norm"], 99) * 1.3, 15.0)))
    ax4.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax4.legend(loc="upper right")

    # ── Panel 5: Training Throughput (Tokens / Sec) ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    # Filter out evaluation step spikes for clean rolling average
    valid_mask = steps["tok_sec"] > 500
    clean_steps = step_nums[valid_mask]
    clean_tok_sec = steps["tok_sec"][valid_mask]
    smoothed_tps = moving_average(clean_tok_sec, window_size=max(10, smooth_window // 2))

    ax5.plot(step_nums, steps["tok_sec"], color=THEME["accent_green"], alpha=0.2, lw=1.0, label="Raw tok/s")
    ax5.plot(clean_steps, smoothed_tps, color=THEME["accent_green"], lw=2.0, label="Steady-State tok/s")
    avg_tps = np.mean(clean_tok_sec)
    ax5.axhline(avg_tps, color=THEME["accent_gold"], linestyle="--", alpha=0.8, label=f"Mean: {avg_tps:,.0f} tok/s")

    ax5.set_title("Training Throughput", fontweight="bold")
    ax5.set_xlabel("Optimizer Steps")
    ax5.set_ylabel("Tokens / Second")
    ax5.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax5.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax5.legend(loc="lower right")

    # ── Panel 6: Cumulative Tokens Processed ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    tokens_millions = steps["tokens"] / 1e6
    ax6.fill_between(step_nums, tokens_millions, color=THEME["accent_blue"], alpha=0.15)
    ax6.plot(step_nums, tokens_millions, color=THEME["accent_blue"], lw=2.0, label="Cumulative Tokens")
    ax6.set_title("Cumulative Token Volume", fontweight="bold")
    ax6.set_xlabel("Optimizer Steps")
    ax6.set_ylabel("Tokens (Millions)")
    ax6.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}M"))
    ax6.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax6.legend(loc="upper left")

    # Save to disk
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\n[Dashboard Saved] High-resolution metric plot saved to -> {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary Report Printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_report(steps: Dict[str, np.ndarray], evals: Dict[str, np.ndarray]):
    """Prints a structured terminal summary of training progression."""
    current_step = steps["step"][-1]
    total_tokens = steps["tokens"][-1]
    initial_loss = steps["loss"][0]
    current_loss = steps["loss"][-1]
    min_loss = np.min(steps["loss"])
    mean_tok_sec = np.mean(steps["tok_sec"][steps["tok_sec"] > 500])

    hr = "=" * 64
    print(f"\n{hr}")
    print("  BELLHART TRAINING SUMMARY & METRICS")
    print(f"{hr}")
    print(f"  Current Step       : {current_step:,}")
    print(f"  Total Tokens       : {total_tokens:,} ({total_tokens / 1e6:.2f} Million)")
    print(f"  Initial Loss       : {initial_loss:.4f}")
    print(f"  Current Loss       : {current_loss:.4f}")
    print(f"  Lowest Train Loss  : {min_loss:.4f}")
    print(f"  Current Learning Rt: {steps['lr'][-1]:.2e}")
    print(f"  Mean Throughput    : {mean_tok_sec:,.0f} tokens/second")

    if len(evals["val_loss"]) > 0:
        best_val_idx = np.argmin(evals["val_loss"])
        best_val = evals["val_loss"][best_val_idx]
        best_step = evals["step"][best_val_idx]
        best_ppl = evals["ppl"][best_val_idx]
        print(f"  Best Validation Loss: {best_val:.4f} (at Step {best_step:,})")
        print(f"  Best Val Perplexity : {best_ppl:.2f}")
    print(f"{hr}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BellHart Training Log Visualizer & Metric Dashboard")
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs.txt",
        help="Path to training log file (default: logs.txt, fallback: logs/training_log.txt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training_plots.png",
        help="Output filepath for the rendered dashboard image (default: training_plots.png)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Rolling average smoothing window for curves (default: 25 steps)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in an interactive window after saving",
    )
    args = parser.parse_args()

    # Determine log file path
    target_log = args.log_file
    if not os.path.exists(target_log):
        if os.path.exists("logs/training_log.txt"):
            target_log = "logs/training_log.txt"
        else:
            print(f"Error: Log file '{target_log}' not found.")
            return

    print(f"Parsing training logs from: {target_log} ...")
    steps, evals = parse_log_file(target_log)
    print_summary_report(steps, evals)
    create_dashboard(steps, evals, output_path=args.out, smooth_window=args.smooth, show=args.show)


if __name__ == "__main__":
    main()

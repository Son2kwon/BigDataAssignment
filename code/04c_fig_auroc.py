"""Phase 5 그림: 시점 N별 test AUROC와 이슈 내 AUROC (B0, B1, M1).

입력: results/phase5/discrimination_compare.csv (04_m1.py 산출물)
출력: results/phase5/figures/auroc_by_n.png
실행: uv run --no-sync python code/04c_fig_auroc.py
표준 라이브러리 csv + matplotlib만 쓴다. 학습·선택은 하지 않는다 (이미 보고한 test 수치를 그림으로만 옮긴다).
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/phase5/discrimination_compare.csv"
OUT = ROOT / "results/phase5/figures/auroc_by_n.png"

MODELS = [  # (model id, label, style)
    ("b0_err", "B0 error count", dict(color="#9a9a9a", ls="--", marker="s")),
    ("b1_n", "B1 issue text", dict(color="#c77c2e", ls="--", marker="^")),
    ("m1_lr", "M1 LR", dict(color="#7aa6d6", ls="-", marker="o")),
    ("m1_xgb", "M1 XGB", dict(color="#2f6db5", ls="-", marker="o")),
    ("m1_xgb_b1", "M1 XGB + B1", dict(color="#123d73", ls="-", marker="D")),
]
PANELS = [("auroc", "Test AUROC"), ("within_pair", "Within-issue AUROC (pair-weighted)")]


def main():
    rows = [r for r in csv.DictReader(SRC.open()) if r["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, PANELS):
        for mid, label, style in MODELS:
            pts = sorted((int(float(r["N"])), r) for r in rows if r["model"] == mid)
            if not pts:
                continue
            ns = [n for n, _ in pts]
            ys = [float(r[col]) for _, r in pts]
            lo = [float(r[col + "_lo"]) if r.get(col + "_lo") else y for (_, r), y in zip(pts, ys)]
            hi = [float(r[col + "_hi"]) if r.get(col + "_hi") else y for (_, r), y in zip(pts, ys)]
            ax.plot(ns, ys, label=label, lw=1.8, ms=5, **style)
            if mid in ("b1_n", "m1_xgb_b1", "m1_xgb"):
                ax.fill_between(ns, lo, hi, color=style["color"], alpha=0.10, lw=0)
        ax.axhline(0.5, color="black", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("N (first N steps observed)")
        ax.set_xticks([3, 5, 8, 10, 15, 20])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC (test, 95% issue bootstrap band)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, metadata={"Software": None})
    print(OUT)


if __name__ == "__main__":
    main()

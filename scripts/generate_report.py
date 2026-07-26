#!/usr/bin/env python
"""F12-01/02: compile scripts/run_full_evaluation.py's results into a
markdown report with tables, ready to paste into the thesis chapter.

Usage:
    .venv/Scripts/python.exe scripts/generate_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_PATH = REPO_ROOT / "experiments" / "full_evaluation_summary.json"
REPORT_PATH = REPO_ROOT / "experiments" / "F8_F11_results_report.md"

DISPLAY_NAMES = {
    "proposed_A3_running_average": "Sistem yang diusulkan (A3 fusion + running-average)",
    "B1_static": "B1 - Static prototype (ablation continual learning)",
    "A1_ecapa_only": "A1 - ECAPA-TDNN saja (ablation fusion)",
    "A2_whisper_only": "A2 - Whisper saja (ablation fusion)",
    "ECAPA_standard_baseline": "Baseline: ECAPA-TDNN standar (closed-set, statis)",
    "ProtoNet_vanilla_baseline": "Baseline: Prototypical Network vanilla (closed-set, statis)",
    "xvector_PLDA_baseline": "Baseline: x-vector + PLDA-lite (closed-set, statis)",
}


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"{SUMMARY_PATH} not found -- run scripts/run_full_evaluation.py first")
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    lines = [
        "# Hasil Evaluasi F8-F11 (Functional-Scale Run)",
        "",
        f"- Threshold terkalibrasi: **{data['calibration_threshold']:.4f}** (EER kalibrasi: {data['calibration_eer']:.4f})",
        f"- Task speakers dengan audio real tersedia: **{data['n_task_speakers_available']}/100** "
        f"di **{data['n_sessions_available']}/10** sesi",
        f"- Pengulangan per konfigurasi: **{data['n_reps']}** (proposal: 10 -- dikurangi untuk skala fungsional)",
        f"- Episode training per model fusion: **{data['n_train_episodes']}**",
        f"- Total waktu eksekusi: **{data['elapsed_minutes']:.1f} menit**",
        "",
        "## Tabel Ringkasan (F9 Ablation + F10 Baseline)",
        "",
        "| Konfigurasi | Accuracy (mean±std) | Forgetting Measure |",
        "|---|---|---|",
    ]
    for key, row in data["summary"].items():
        name = DISPLAY_NAMES.get(key, key)
        fm = f"{row['mean_forgetting']:.4f}" if row["mean_forgetting"] is not None else "-"
        lines.append(f"| {name} | {row['mean_accuracy']:.3f} ± {row['std_accuracy']:.3f} | {fm} |")

    lines += [
        "",
        "## Uji Signifikansi Statistik (F11)",
        "",
        "| Perbandingan | Uji | p-value | Signifikan (Bonferroni-corrected)? |",
        "|---|---|---|---|",
    ]
    for key, row in data["statistics"].items():
        a, b = key.split("_vs_")
        lines.append(
            f"| {DISPLAY_NAMES.get(a, a)} vs {DISPLAY_NAMES.get(b, b)} | {row['test_used']} | "
            f"{row['p_value']:.4f} | {'Ya' if row['significant'] else 'Tidak'} |"
        )

    lines += [
        "",
        "## Catatan Konfigurasi (fusi beku / residual-init)",
        "",
        "Angka-angka di atas dihasilkan dari mekanisme **asli** (kalibrasi EER, "
        "harness FSCIL 10-sesi, uji statistik) di atas **audio VoxCeleb1/2 asli** "
        "untuk `task_speakers`/`episodic_sessions` di `data/splits/full_split.json` "
        "-- bukan data substitusi.",
        "",
        "Lapisan Gated Attention Fusion dijalankan dengan **inisialisasi residual "
        "(ECAPA-preserving) dan dibekukan (`N_TRAIN_EPISODES=0`)**. Sebuah sweep "
        "jumlah pelatihan (`scripts/sweep_training.py`) menunjukkan fine-tuning "
        "episodik pada skala base kecil ini (71 speaker) **secara monoton menurunkan** "
        "akurasi: fusi residual beku mencapai 0.783 closed-set / 0.734 open-set, "
        "sedangkan setelah 500 episode turun ke 0.276 / 0.187. Penyebabnya, 71 speaker "
        "jauh terlalu sedikit untuk memperbaiki ruang embedding ECAPA yang sudah "
        "dilatih penuh -- pelatihan hanya meng-*overfit* dan merusaknya. Membekukan "
        "fusi menjadikan kualitas ECAPA sebagai batas bawah yang terjamin.",
        "",
        "Konsekuensi yang perlu dilaporkan jujur: karena fusi didominasi ECAPA "
        "(gate ~0.98), A3 (fusi) praktis identik dengan A1 (ECAPA-saja) pada skala ini "
        "-- kontribusi *fusi* belum terbukti di sini; nilai yang terbukti adalah "
        "kerangka **open-set + continual** (B2 running-average unggul atas B1 static, "
        "dan Forgetting Measure mendekati nol). Apakah fusi benar-benar menambah nilai "
        "hanya bisa diuji ulang dengan base training berskala penuh (VoxCeleb2 penuh, "
        "ratusan-ribuan speaker) sebagai kerja lanjutan; N_REPS juga dikurangi dari 10 "
        "ke " + str(data.get("n_reps", "?")) + " untuk menyelesaikan run dalam satu sesi.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

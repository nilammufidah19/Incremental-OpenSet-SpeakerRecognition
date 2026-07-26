# Hasil Evaluasi F8-F11 (Functional-Scale Run)

- Threshold terkalibrasi: **0.9266** (EER kalibrasi: 0.2193)
- Task speakers dengan audio real tersedia: **99/100** di **10/10** sesi
- Pengulangan per konfigurasi: **5** (proposal: 10 -- dikurangi untuk skala fungsional)
- Episode training per model fusion: **0**
- Total waktu eksekusi: **2.6 menit**

## Tabel Ringkasan (F9 Ablation + F10 Baseline)

| Konfigurasi | Accuracy (mean±std) | Forgetting Measure |
|---|---|---|
| Sistem yang diusulkan (A3 fusion + running-average) | 0.731 ± 0.021 | 0.0004 |
| B1 - Static prototype (ablation continual learning) | 0.666 ± 0.024 | 0.0128 |
| A1 - ECAPA-TDNN saja (ablation fusion) | 0.731 ± 0.021 | 0.0004 |
| A2 - Whisper saja (ablation fusion) | 0.171 ± 0.015 | 0.0111 |
| Baseline: ECAPA-TDNN standar (closed-set, statis) | 0.788 ± 0.017 | 0.0530 |
| Baseline: Prototypical Network vanilla (closed-set, statis) | 0.405 ± 0.017 | 0.1214 |
| Baseline: x-vector + PLDA-lite (closed-set, statis) | 0.264 ± 0.014 | 0.1199 |

## Uji Signifikansi Statistik (F11)

| Perbandingan | Uji | p-value | Signifikan (Bonferroni-corrected)? |
|---|---|---|---|
| Sistem yang diusulkan (A3 fusion + running-average) vs A1 - ECAPA-TDNN saja (ablation fusion) | degenerate_identical_diffs | 1.0000 | Tidak |
| Sistem yang diusulkan (A3 fusion + running-average) vs A2 - Whisper saja (ablation fusion) | paired_t_test | 0.0000 | Ya |
| B1 - Static prototype (ablation continual learning) vs Sistem yang diusulkan (A3 fusion + running-average) | paired_t_test | 0.0037 | Ya |
| Sistem yang diusulkan (A3 fusion + running-average) vs Baseline: ECAPA-TDNN standar (closed-set, statis) | paired_t_test | 0.0002 | Ya |
| Sistem yang diusulkan (A3 fusion + running-average) vs Baseline: Prototypical Network vanilla (closed-set, statis) | paired_t_test | 0.0000 | Ya |
| Sistem yang diusulkan (A3 fusion + running-average) vs Baseline: x-vector + PLDA-lite (closed-set, statis) | paired_t_test | 0.0000 | Ya |

## Catatan Konfigurasi (fusi beku / residual-init)

Angka-angka di atas dihasilkan dari mekanisme **asli** (kalibrasi EER, harness FSCIL 10-sesi, uji statistik) di atas **audio VoxCeleb1/2 asli** untuk `task_speakers`/`episodic_sessions` di `data/splits/full_split.json` -- bukan data substitusi.

Lapisan Gated Attention Fusion dijalankan dengan **inisialisasi residual (ECAPA-preserving) dan dibekukan (`N_TRAIN_EPISODES=0`)**. Sebuah sweep jumlah pelatihan (`scripts/sweep_training.py`) menunjukkan fine-tuning episodik pada skala base kecil ini (71 speaker) **secara monoton menurunkan** akurasi: fusi residual beku mencapai 0.783 closed-set / 0.734 open-set, sedangkan setelah 500 episode turun ke 0.276 / 0.187. Penyebabnya, 71 speaker jauh terlalu sedikit untuk memperbaiki ruang embedding ECAPA yang sudah dilatih penuh -- pelatihan hanya meng-*overfit* dan merusaknya. Membekukan fusi menjadikan kualitas ECAPA sebagai batas bawah yang terjamin.

Konsekuensi yang perlu dilaporkan jujur: karena fusi didominasi ECAPA (gate ~0.98), A3 (fusi) praktis identik dengan A1 (ECAPA-saja) pada skala ini -- kontribusi *fusi* belum terbukti di sini; nilai yang terbukti adalah kerangka **open-set + continual** (B2 running-average unggul atas B1 static, dan Forgetting Measure mendekati nol). Apakah fusi benar-benar menambah nilai hanya bisa diuji ulang dengan base training berskala penuh (VoxCeleb2 penuh, ratusan-ribuan speaker) sebagai kerja lanjutan; N_REPS juga dikurangi dari 10 ke 5 untuk menyelesaikan run dalam satu sesi.
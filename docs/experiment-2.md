# Experiment 2 — Menaikkan Kontribusi Whisper

**Sub-eksperimen 2a:** `exp2a_scorefusion_L4` — ✅ **SUDAH DIJALANKAN**
**Sub-eksperimen 2b/2c:** 🟡 masih rencana
**Batasan tetap:** strict 1-shot (K_SHOT = 1), bebas-pelatihan (mengikuti temuan Experiment 1).
**Hasil utama 2a:** sistem usulan **Accuracy = 0.733 ± 0.017**, Forgetting **0.0004** — **≈ Experiment 1 (0.731), tidak ada peningkatan berarti.**
**Temuan terpenting Experiment 2:** kontribusi Whisper terbukti **kecil** (plafon komplementaritas terbatas); **lever sesungguhnya untuk melampaui baseline adalah THRESHOLD**, bukan fusi.

Versi HTML: [`experiment-2.html`](experiment-2.html). Basis: [Experiment 1](experiment-1.md) (frozen residual fusion). Dokumen ini melaporkan apa yang dicoba, diimplementasikan, dan hasil aktualnya.

---

## 0. Ringkasan Eksekutif

| | Experiment 1 | **Experiment 2a** |
|---|---|---|
| Mekanisme fusi | Gated **embedding** fusion (beku, residual-init) | **Score-level** fusion (gabung jarak) |
| Fitur Whisper | Lapisan tengah (L3), mean-pool | **Lapisan 4 (L4)**, mean-pool |
| Bobot | gate ~0.98 (didominasi ECAPA) | w = 0.4 pada ECAPA (tetap) |
| Akurasi sistem usulan | 0.731 | **0.733** *(±0.017)* |
| Forgetting | 0.0004 | 0.0004 |
| Kesimpulan | — | **Whisper tak menambah nilai berarti; threshold jadi tersangka utama** |

**Tiga hasil kunci Experiment 2:**
1. **Plafon komplementaritas Whisper kecil** (+2.6% L3 → +4.0% L4 pada oracle). Whisper jarang benar saat ECAPA salah.
2. **Score-level fusion tidak mengungguli Experiment 1** pada metrik open-set (0.733 ≈ 0.731). Gain +1.4% yang terlihat di closed-set snapshot **tidak bertahan**.
3. **Threshold EER adalah sumber kerugian utama** dan bisa diperbaiki — memicu Experiment 3.

---

## 1. Motivasi — gap dari Experiment 1

Experiment 1 menaikkan sistem usulan 0.235 → 0.731 tetapi menyisakan:
1. **Kontribusi fusi Whisper = 0** (gate ~0.98 → A3 ≡ A1, p=1.0).
2. **Belum melampaui baseline** ECAPA closed-set (0.788).

Tujuan Experiment 2: membuat kontribusi Whisper terukur & positif — tanpa menaikkan shot, tanpa fine-tuning yang merusak (temuan Experiment 1).

---

## 2. Yang dicoba & diimplementasikan (2a)

### 2.1 Perubahan arsitektur vs Experiment 1

**a. Fitur Whisper → lapisan encoder 4** (bukan lapisan tengah L3).
Sweep bebas-pelatihan menemukan L4 paling diskriminatif. Di-cache terpisah sebagai backbone `whisper_l4` (`layer_fraction = 4/6`) sehingga koeksis dengan varian lama.

**b. Fusi → score-level (late fusion).**
Alih-alih memadukan embedding, keputusan dibuat di level **jarak**:

```
skor(query, speaker_k) = w · d_ecapa(q, c_k) + (1 − w) · d_whisper(q, c_k)
```

dengan `d_*` = cosine distance di ruang asli tiap backbone (w = 0.4 pada ECAPA).

**Trik implementasi (tanpa mesin baru):** score fusion di atas ekuivalen dengan nearest-neighbor Euclidean pada **konkatenasi berbobot**

```
out = [ √w · ê_ecapa ; √(1−w) · ê_whisper ]
```

karena jarak-Euclidean-kuadrat antar dua vektor semacam itu = `2·(w·cosdist_ecapa + (1−w)·cosdist_whisper)` — persis score fusion (monoton). Diverifikasi numerik eksak. Direalisasikan sebagai modul **`ScoreFusionEmbed`** (bebas-parameter) yang menggantikan `GatedAttentionFusion` — sehingga sistem prototype-distance, kalibrasi threshold, dan harness FSCIL **berjalan tanpa perubahan**. Mode ablasi memetakan ke bobot tetap: `ecapa_only → w=1`, `whisper_only → w=0`, `fusion → w=0.4`.

### 2.2 File yang diubah

| File | Perubahan |
|---|---|
| `src/models/fusion.py` | tambah `ScoreFusionEmbed` (bebas-parameter, konkatenasi berbobot) |
| `src/features/cache.py` | varian backbone `whisper_l4` (namespace cache terpisah) |
| `src/prototypical/data.py` | `build_raw_embedding_index(..., whisper_backbone=...)` |
| `src/system.py` | parameter `whisper_backbone` |
| `src/experiments.py` | field `fusion_strategy`, `score_fusion_weight`, `whisper_backbone`; tag `exp2a_scorefusion_L4` |
| `scripts/run_full_evaluation.py` | cabang strategi fusi (embedding vs score) dari `ExperimentConfig` |

---

## 3. Eksplorasi pra-eksperimen (dasar keputusan)

### 3.1 Plafon komplementaritas (`complementarity_ceiling`)
FSCIL closed-set, backbone mentah:

| | Accuracy | vs ECAPA |
|---|---|---|
| ECAPA saja | 0.853 | — |
| Whisper saja | 0.290 | −0.563 |
| Oracle (salah satu benar) | 0.879 | **+0.026** |
| Score-fusion terbaik (w=0.5) | 0.861 | +0.008 |

→ Whisper jarang mengoreksi ECAPA; plafon kecil.

### 3.2 Sweep ekstraksi Whisper (`exp2_whisper_sweep`)
Snapshot 100-way, 1-shot; ECAPA-only = 0.756:

| Config | Whisper-only | Oracle (plafon) | Score-fusion terbaik |
|---|---|---|---|
| L3/mean *(≈produksi lama)* | 0.208 | 0.778 | 0.764 |
| **L4/mean** | **0.232** | **0.796 (+0.040)** | **0.770 (+0.014, w=0.4)** |
| L4/mean_std | 0.192 | 0.796 | 0.766 |

→ L4 > L3; mean+std tak membantu; score-fusion menangkap +1.4% (closed-set snapshot) yang gated-fusion lewatkan. **Dasar** untuk memilih 2a: L4 + score-fusion, w=0.4.

---

## 4. Hasil Experiment 2a (run resmi)

Konfigurasi: `exp2a_scorefusion_L4`, 5 repetisi, audio VoxCeleb nyata, threshold terkalibrasi = **0.5967** (EER **0.2138**). Waktu eksekusi 103 menit (menghitung embedding Whisper-L4 dari awal).

| Konfigurasi | Accuracy (mean ± std) | Forgetting |
|---|---|---|
| **Sistem usulan (A3, score-fusion w=0.4)** | **0.733 ± 0.017** | **0.0004** |
| B1 — Static prototype | 0.675 ± 0.019 | 0.0123 |
| A1 — ECAPA saja (w=1.0) | 0.069 ± 0.017 ⚠️ *(artefak — lihat §5)* | 0.0 |
| A2 — Whisper saja (w=0.0) | 0.200 ± 0.016 ⚠️ *(artefak)* | 0.0133 |
| Baseline: ECAPA standar (closed-set) | 0.788 ± 0.017 | 0.0530 |
| Baseline: ProtoNet vanilla | 0.405 ± 0.017 | 0.1214 |
| Baseline: x-vector + PLDA-lite | 0.264 ± 0.014 | 0.1199 |

**Signifikansi (α = 0.00833):** usulan vs B1 signifikan (p=0.0019, continual membantu); usulan vs ECAPA baseline signifikan (p=0.0005, **baseline masih unggul**); usulan vs ProtoNet & Whisper-only signifikan; usulan vs x-vector **tidak** signifikan (Wilcoxon p=0.0625).

> **Kesimpulan 4:** A3 score-fusion (0.733) ≈ Experiment 1 (0.731). Menaikkan lapisan Whisper ke L4 dan memindah ke fusi skor **tidak** menghasilkan peningkatan berarti pada metrik open-set FSCIL.

---

## 5. Dua masalah yang terungkap

### 5.1 Artefak threshold pada ablasi (bug harness)
Threshold dikalibrasi pada model A3 (w=0.4) lalu dipakai bersama untuk A1/A2. Score-fusion mengubah **skala jarak** per-w, sehingga threshold w=0.4 salah total untuk A1 (w=1.0) → A1 anjlok ke 0.069 (artefak, bukan masalah embedding). Perbaikan: kalibrasi threshold **per-konfigurasi**. (Tidak memengaruhi angka A3 yang thresholdnya benar.)

### 5.2 Threshold EER = sumber kerugian utama (studi threshold)
Sweep bobot w dengan threshold dikalibrasi ulang **per-w**, tiga regime (`exp2a_threshold_study`, 3 seed):

| w (ECAPA) | closed (∞) | open @EER | open @p95 | EER |
|---|---|---|---|---|
| 0.0 (Whisper murni) | 0.204 | 0.137 | 0.197 | 0.370 |
| 0.3 | 0.746 | 0.747 | 0.750 | 0.217 |
| 0.5 | 0.745 | 0.749 | 0.759 | 0.218 |
| **0.6** | **0.749** | 0.747 | **0.761** | 0.218 |
| 1.0 (ECAPA murni) | 0.744 | 0.746 | 0.757 | 0.220 |

- **`open@p95 ≈ closed`** → dengan threshold permisif (persentil-95 genuine, ~5% false-reject) penalti open-set **hampir hilang**; EER menolak terlalu banyak genuine.
- **Metrik FSCIL tidak punya query unknown** → setiap penolakan = false-reject murni; EER (menyeimbangkan FAR=FRR) adalah titik operasi yang salah untuk metrik ini.
- **Whisper marginal:** w=0.6 terbaik hanya +0.004 vs ECAPA murni.

→ Memicu **Experiment 3 (fokus threshold)**.

---

## 6. Analisis & kesimpulan

1. **Whisper bukan jalan menembus baseline.** Plafon komplementaritas kecil (+4% oracle), dan pada metrik open-set kontribusinya ~0. Lapisan L4 & fusi skor tak mengubah kesimpulan.
2. **Threshold adalah lever nyata.** EER menghukum sistem usulan (~0.01–0.05) tanpa manfaat pada metrik tanpa-unknown. Titik operasi low-FRR (p95) memulihkan hampir semua kerugian; `open@p95 ≈ closed`.
3. **Forgetting tetap ~0** dan continual tetap membantu (B2 > B1, p=0.0019) — kekuatan kerangka bertahan.

**Rekomendasi ke Experiment 3:** (a) ganti titik operasi EER → low-FRR (`calibration_target_frr`), (b) laporkan **akurasi closed-set** berdampingan (perbandingan apple-to-apple dengan baseline closed-set), (c) opsional AS-Norm untuk menurunkan EER 0.22. Ini peluang paling nyata menyamai/melampaui baseline secara jujur.

---

## 7. Status sub-eksperimen

| Tag | Status | Hasil |
|---|---|---|
| `exp2a_scorefusion_L4` | ✅ dijalankan | 0.733 (≈ exp1; Whisper tak menambah) |
| `exp2b_whisper_small` | 🟡 rencana | model lebih besar (belum) |
| `exp2c_multilayer` | 🟡 rencana | agregasi L3–L5 (belum) |

> **Catatan:** setelah eksplorasi 2a, prioritas bergeser dari 2b/2c ke **Experiment 3 (threshold)**, karena data menunjukkan Whisper bukan lever utama. 2b/2c tetap terdaftar bila plafon perlu diuji lebih jauh.

---

## 8. Feature-flag & reproduksi

Tag `exp2a_scorefusion_L4` terdaftar di `src/experiments.py` (field `fusion_strategy="score"`, `score_fusion_weight=0.4`, `whisper_backbone="whisper_l4"`).

```powershell
# menjalankan Experiment 2a
$env:ACTIVE_EXPERIMENT = "exp2a_scorefusion_L4"
.venv\Scripts\python.exe scripts\run_full_evaluation.py
```

> **Catatan artefak pelaporan:** run 2a menimpa `experiments/full_evaluation_summary.json` (termasuk artefak A1=0.069). Karena 2a inkonklusif, **Experiment 1 tetap hasil headline**; summary/report sebaiknya dipulihkan ke exp1 (jalankan ulang tag `exp1_frozen_residual`, cepat karena cache sudah panas).

---

*Laporan Experiment 2. Terkait: [experiment-1.md](experiment-1.md) · [README.md](README.md) · [experiment-2.html](experiment-2.html) · `src/experiments.py`.*

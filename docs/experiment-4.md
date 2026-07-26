# Experiment 4 — Gerbang Keputusan Fusi: Plafon Komplementaritas di Ruang AS-Norm

**Status:** ✅ **SUDAH DIJALANKAN** (18 Juli 2026)
**Bentuk:** analisis gerbang keputusan (`scripts/exp4_complementarity_asnorm.py`) — **bukan run resmi**; task speaker resmi & paruh-deteksi tidak disentuh.
**Batasan tetap:** strict 1-shot (K_SHOT = 1), bebas-pelatihan, basis sistem = [`exp3b_asnorm`](experiment-3.md) (AS-Norm c300/k200).
**Versi HTML (laporan lengkap + arsitektur):** [`experiment-4.html`](experiment-4.html)

> ## 🧭 Verdik utama
> **Fusi ECAPA+Whisper untuk identifikasi resmi DITUTUP — kini dengan bukti di *kedua* ruang skor (mentah & AS-Norm).**
> Plafon oracle di ruang AS-Norm hanya **+0.0225 ± 0.0054** (whisper_l4) / **+0.0133 ± 0.0042** (whisper L3), dan
> **score-fusion nyata tidak menangkap sedikit pun dari plafon itu**: sweep bobot w terbaik (w=0.95) memberi **0.8408 — masih di bawah ECAPA-saja (0.8417)** → **gerbang G4.2 GAGAL**.
> Konsekuensi: jalur multi-backbone tesis pindah ke **[Experiment 5](experiment-5.md) (ganti backbone)**.
> Temuan sekunder (4c): aturan deteksi dua-ruang `mean` menurunkan EER validasi 0.1654 → **0.1508** dan menaikkan AUROC 0.909 → **0.924** — sinyal awal "saksi kedua", tapi baru diukur pada 40 query unknown → dibawa sebagai kandidat ke run resmi Experiment 5, bukan ke produksi sekarang.

---

## 1. Pertanyaan riset & jawaban

| RQ | Pertanyaan | Jawaban |
|---|---|---|
| RQ4.1 | Berapa plafon oracle ECAPA∪Whisper di ruang AS-Norm exp3b? | **+0.0225** (whisper_l4) / +0.0133 (L3) — AS-Norm **tidak mengubah plafon** (mentah: +0.0208 / +0.0125); error yang diperbaiki AS-Norm ternyata *bukan* error yang bisa dikoreksi Whisper |
| RQ4.2 | Apakah score-fusion ternormalisasi menangkap plafon itu? | **Tidak.** Akurasi fusi monoton naik menuju w→1 dan tak pernah melewati ECAPA-saja; plafon oracle butuh *selector per-query* yang sempurna, yang tidak tersedia bebas-pelatihan |
| RQ4.3 | Apakah disagreement dua-backbone membantu **deteksi unknown**? | **Indikasi ya** (aturan `mean`: EER −0.015, AUROC +0.015), tetapi TAR@1%FAR turun (0.711→0.681) dan n unknown hanya 40 → **belum konklusif** |

---

## 2. Metodologi (yang benar-benar dijalankan)

- **Data:** task validasi = 100 speaker pertama paruh-validasi `reserved_unknown_pool` (protokol anti-overfit exp3 §4; 108 usable dari 110), 1-shot support + 4 query/speaker = 400 query genuine per seed, 3 seed (0/1/2). 8 speaker sisa (40 utterance) menjadi query unknown 4c. Paruh-deteksi (111 speaker) tak disentuh.
- **Ruang skor per backbone:** ruang **natif** masing-masing (embedding mentah L2-norm; Euclidean ≡ cosine monoton — konvensi `ScoreFusionEmbed`), *bukan* proyeksi gated-fusion — residual-init hanya melindungi sisi ECAPA, proyeksi acak merusak ruang Whisper (pelajaran exp1), jadi ruang natif adalah pengukuran Whisper yang paling adil.
- **AS-Norm per ruang:** cohort = **300 utterance base_train yang sama** untuk kedua ruang (round-robin antar speaker, disiplin `build_cohort`), top_k = 200 (nilai terkunci exp3b), via kelas `ASNorm` produksi tanpa modifikasi.
- **Prototype statis** (snapshot 1-shot, tanpa continual update): plafon adalah properti geometri identifikasi; dinamika continual hanya diuji di run resmi.
- **Dua varian Whisper** dianalisis: `whisper_l4` (terbaik exp2) dan `whisper` (L3, yang dipakai exp1/exp3).
- Run pertama menghitung 540 embedding `whisper_l4` paruh-validasi yang belum pernah di-cache (~13 menit, GPU RTX 3050); analisis dengan cache panas: **0.2 menit**.

---

## 3. Hasil

### 3.1 Exp4a — plafon oracle (mean ± std, 3 seed, 400 query/seed)

| Varian | Ruang | ECAPA saja | Whisper saja | Oracle | **Δ plafon** |
|---|---|---|---|---|---|
| whisper_l4 | mentah | 0.8117 | 0.2167 | 0.8325 | +0.0208 ± 0.0101 |
| whisper_l4 | **AS-Norm** | 0.8417 | 0.2425 | 0.8642 | **+0.0225 ± 0.0054** |
| whisper (L3) | mentah | 0.8117 | 0.2300 | 0.8242 | +0.0125 ± 0.0074 |
| whisper (L3) | **AS-Norm** | 0.8350 | 0.2317 | 0.8483 | **+0.0133 ± 0.0042** |

Hipotesis "AS-Norm membuka ruang komplementaritas baru" **tertolak** — plafon praktis identik di kedua ruang. AS-Norm menaikkan ECAPA (+0.030 di snapshot ini, konsisten temuan hubness exp3) tetapi kasus "Whisper benar & ECAPA salah" tetap langka: hanya **11 dari 63 error ECAPA** (seed 0, whisper_l4).

### 3.2 Exp4b — sweep score-fusion `w·z_ecapa + (1−w)·z_whisper` (ruang AS-Norm)

| w (ECAPA) | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | **ECAPA saja** |
|---|---|---|---|---|---|---|---|
| whisper_l4 | 0.6992 | 0.7450 | 0.7942 | 0.8317 | 0.8367 | 0.8408 | **0.8417** |
| whisper (L3) | 0.6875 | 0.7325 | 0.7875 | 0.8183 | 0.8275 | 0.8325 | **0.8350** |

Monoton naik menuju w→1: **setiap campuran Whisper hanya menambah noise**. (Di ruang mentah w=0.7 sempat +0.005 di atas ECAPA — replikasi gain kecil closed-set exp2 — tetapi tidak bertahan di ruang AS-Norm yang dipakai produksi.)

### 3.3 Exp4c — deteksi unknown dua-ruang (validasi; 400 genuine vs **40 unknown**)

| Aturan (whisper_l4) | EER | AUROC | TAR@1%FAR |
|---|---|---|---|
| `a_only` (ECAPA, baseline) | 0.1654 ± 0.0296 | 0.9090 | 0.7108 |
| `mean` (½(sₑ+s_w)) | **0.1508 ± 0.0209** | **0.9244** | 0.6808 |
| `a_disagree` (sₑ + 0.5·[argmin beda]) | 0.1775 | 0.9068 | 0.7058 |
| `b_only` / `max` | 0.2842 / 0.2675 | 0.7853 / 0.8001 | 0.4050 / 0.4233 |

---

## 4. Gerbang keputusan — hasil evaluasi

| Gerbang | Kriteria | Hasil | Verdik |
|---|---|---|---|
| **G4.1** | Δ plafon AS-Norm < +0.02 → tutup fusi | whisper_l4: +0.0225 (nominal *lolos ambang*, tapi hanya 0.5σ di atasnya); L3: +0.0133 (di bawah) | ambigu → diputuskan oleh G4.2 |
| **G4.2** | ada w dengan acc fusi > ECAPA-saja | **Tidak ada** — terbaik 0.8408 < 0.8417 (l4), 0.8325 < 0.8350 (L3) | ❌ **GAGAL → fusi Whisper DITUTUP**; tag produksi `exp4b_scorefusion_asnorm` **tidak dibuat** |
| **G4.3** | aturan deteksi menurunkan EER ≥ 0.01 | `mean`: −0.0146 EER, +0.0154 AUROC; tapi TAR@1% −0.030 dan n=40 | ⚠️ lolos nominal, **underpowered** → dibawa sebagai kandidat evaluasi di run resmi Experiment 5 |

**Keputusan akhir:** tidak ada perubahan jalur produksi; `exp3b_asnorm` tetap konfigurasi aktif. Narasi multi-backbone tesis berlanjut ke [Experiment 5](experiment-5.md) (backbone pengganti Whisper) — prasyaratnya kini terpenuhi dengan bukti terukur dua-ruang.

---

## 5. Perubahan kode (semua ADITIF — tanpa menyentuh jalur produksi)

Disiplin feature-flag exp3 dipertahankan dengan cara paling ketat: karena G4.2 gagal, **tidak ada field `ExperimentConfig` baru dan tidak ada tag baru** — perilaku setiap tag lama (`baseline_v0` … `exp3c_dualmetric`) tak berubah satu bit pun. Revert Experiment 4 = tidak menjalankan script-nya.

| File | Status | Isi |
|---|---|---|
| `src/evaluation/complementarity.py` | **baru** | helper analisis murni-numpy: `l2_normalize`, `oracle_report`, `fusion_accuracy`, `detection_scores` — dipakai ulang untuk screening kandidat Experiment 5 |
| `scripts/exp4_complementarity_asnorm.py` | **baru** | driver analisis 4a+4b+4c (pola `exp3_validation_sweep.py`), output `experiments/exp4_ceiling_asnorm.json` |
| `tests/test_experiment4.py` | **baru** | 7 unit test (oracle/fusi/deteksi + invariansi skala AS-Norm per-ruang); 11 test exp3 tetap lolos |
| `src/experiments.py`, `manager.py`, `calibration.py`, dst. | **tak diubah** | jalur keputusan produksi identik dengan exp3b |

---

## 6. Reproduksi

```powershell
# analisis lengkap 4a + 4b + 4c (kedua varian Whisper, 3 seed)
.venv\Scripts\python.exe scripts\exp4_complementarity_asnorm.py
# hasil: experiments\exp4_ceiling_asnorm.json (+ log experiments\exp4_run_log.txt)

# unit test
.venv\Scripts\python.exe -m pytest tests\test_experiment4.py -q
```

---

*Laporan Experiment 4. Terkait: [experiment-2.md](experiment-2.md) · [experiment-3.md](experiment-3.md) · [experiment-5.md](experiment-5.md) · [experiment-4.html](experiment-4.html) · [README.md](README.md).*

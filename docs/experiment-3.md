# Experiment 3 — Pembenahan Threshold Open-Set

**Status:** ✅ **SUDAH DIJALANKAN** (11 Juli 2026)
**Tag:** `exp3a_lowfrr` ✅ · `exp3b_asnorm` ✅ · `exp3c_dualmetric` ✅ *(= run pelaporan exp3b)*
**Batasan tetap:** strict 1-shot (K_SHOT = 1), bebas-pelatihan (0 episode, mengikuti Experiment 1).
**Basis:** [Experiment 1](experiment-1.md) (frozen residual fusion, 0.731). Whisper terbukti bukan lever ([Experiment 2](experiment-2.md)) → fokus threshold.

> ## 🏆 Hasil utama
> **Sistem usulan (exp3b, AS-Norm + low-FRR): Accuracy = 0.865 ± 0.008** — **melampaui baseline ECAPA closed-set (0.788 ± 0.017) secara signifikan** (paired t-test p = 0.0015 < α_Bonferroni = 0.00833), dengan **Forgetting 0.0022 (~0)**, **deteksi unknown EER 0.111** (target ≤ 0.15 ✓), **AUROC 0.952**, **TAR@FAR=1% 0.594**.
> Decision gate *"melampaui baseline"* (open-set Acc > 0.788 signifikan) **TERCAPAI** — bahkan stretch goal terlampaui (+0.077 absolut di atas baseline; +0.134 di atas exp1).

Versi HTML: [`experiment-3.html`](experiment-3.html). Angka rancangan/perkiraan pada versi lama dokumen ini sudah diganti hasil run resmi.

---

## 1. Diagnosis — kenapa threshold jadi masalah (rekap rancangan)

Dari studi threshold Experiment 2 (sweep bobot, threshold dikalibrasi ulang per-w, 3 seed):

| w (ECAPA) | closed (∞) | open @EER | open @p95 | EER |
|---|---|---|---|---|
| 0.5 | 0.745 | 0.749 | 0.759 | 0.218 |
| 0.6 | 0.749 | 0.747 | **0.761** | 0.218 |
| 1.0 (ECAPA murni) | 0.744 | 0.746 | **0.757** | 0.220 |

1. **`open@p95 > open@EER`** — titik EER menolak terlalu banyak genuine; metrik FSCIL tidak punya query unknown sehingga setiap penolakan = false-reject murni.
2. **EER tinggi (0.22)** — separasi genuine/impostor buruk pada 1-shot.
3. **Akar masalah:** titik operasi dan cara skoring tidak dioptimalkan untuk metrik yang dipakai.

Masalah yang dibenahi: **P1** titik operasi EER salah → low-FRR (3a); **P2** EER tinggi → AS-Norm (3b); **P3** satu angka menyembunyikan kualitas → dual-metric (3c); **P4** threshold per-ablasi salah pakai → kalibrasi per-konfigurasi; **P5** drift `running_average` → terjawab lewat temuan 3a/3b (§6.3).

---

## 2. CATATAN PERUBAHAN LENGKAP (changelog implementasi)

Semua perubahan **di belakang feature flag** — field baru `ExperimentConfig` dengan default yang mempertahankan perilaku lama persis; tag `baseline_v0`/`exp1_frozen_residual`/`exp2a_scorefusion_L4` tidak berubah perilakunya.

### 2.1 Field `ExperimentConfig` baru (`src/experiments.py`)

| Field baru | Default (perilaku lama) | Nilai exp3a | Nilai exp3b/3c | Fungsi |
|---|---|---|---|---|
| `calibration_strategy` | `"eer"` | `"target_frr"` | `"target_frr"` | strategi pemilihan titik operasi threshold |
| `target_frr` | 0.05 *(tak terpakai saat "eer")* | **0.01** *(dikunci via validasi)* | **0.05** *(dikunci via validasi)* | FRR target: threshold = kuantil-(1−FRR) jarak genuine |
| `per_config_calibration` | `False` | `True` | `True` | **fix P4**: kalibrasi threshold per model fusi (A1/A2/A3), bukan memakai threshold A3 untuk semua |
| `score_norm` | `"none"` | `"none"` | `"asnorm"` | normalisasi skor sebelum keputusan threshold |
| `asnorm_cohort_size` | 300 | — | **300** | jumlah utterance cohort (dari base_train) |
| `asnorm_top_k` | 100 | — | **200** *(dikunci via validasi)* | ukuran subset adaptif statistik cohort |
| `report_open_set_detection` | `False` | `True` | `True` | jalur query unknown + metrik deteksi + akurasi closed-set |

### 2.2 File yang diubah / dibuat

| File | Perubahan |
|---|---|
| `src/prototypical/calibration.py` | + `find_operating_point(genuine, impostor, strategy, target_frr)`; `CalibrationResult` diperluas (`strategy`, `target_frr`, `far/frr_at_threshold`); `calibrate_threshold(..., strategy, target_frr, score_normalizer)`; `build_genuine_impostor_distances(..., score_normalizer)` — kalibrasi berlangsung di ruang skor yang sama dengan keputusan runtime |
| `src/prototypical/score_norm.py` **(baru)** | AS-Norm simetris adaptif: `d_norm = ½[(d−μ_topK(q→cohort))/σ_topK(q→cohort) + (d−μ_topK(p→cohort))/σ_topK(p→cohort)]`, orientasi tetap jarak (kecil = genuine); `build_cohort()` sampling round-robin antar-speaker base_train |
| `src/continual/manager.py` | param `score_normalizer` (keputusan di ruang ternormalisasi bila di-set); field baru `ContinualProcessResult.nearest_speaker_id` (prediksi closed-set, apapun keputusan reject); method `score_sample()` — keputusan **tanpa efek samping** (tanpa update prototype, tanpa buffer novelty) untuk query unknown |
| `src/system.py` | meneruskan `score_normalizer`; method `score()` (jalur bebas-efek-samping) |
| `src/evaluation/baseline_system.py` | `SingleBackboneSystem.score()` (kesimetrian antarmuka) |
| `src/evaluation/metrics.py` | + `auroc()` (formulasi Mann-Whitney U, eksak) dan `tar_at_far()` (threshold = kuantil FAR impostor) |
| `src/evaluation/fscil.py` | + `run_fscil_detailed()` → `FSCILResult` (open-set + closed-set per sesi/task + jarak keputusan genuine sesi-final + jarak query unknown); `run_fscil()` lama tetap ada & memakai jalur yang sama (kontrak tak berubah) |
| `src/data/splits.py` | + `split_reserved_pool_halves()` — pemisah deterministik reserved_unknown_pool menjadi paruh-validasi & paruh-deteksi (penjaga anti-leakage tunggal, dipakai kedua konsumen) |
| `src/experiments.py` | 7 field baru (§2.1) + tag `exp3a_lowfrr`, `exp3b_asnorm`, `exp3c_dualmetric` |
| `scripts/run_full_evaluation.py` | kalibrasi per-konfigurasi (P4); strategi & normalizer dari config; cohort AS-Norm per model fusi; jalur unknown query (530 utterance, 106 speaker paruh-deteksi); pelaporan closed-set + EER/AUROC/TAR@FAR per konfigurasi; summary per-eksperimen `full_evaluation_summary_<tag>.json` agar run berikutnya tak menimpa hasil (pelajaran dari Experiment 2 §8) |
| `scripts/exp3_validation_sweep.py` **(baru)** | sweep validasi anti-overfit (§4) |
| `tests/test_experiment3.py` **(baru)** | 11 unit test: target-FRR, AS-Norm (orientasi, separasi skala-bergeser, cohort round-robin, hook manager), AUROC/TAR@FAR, `score_sample` bebas-efek-samping |

### 2.3 Perbaikan insidental (ditemukan selama Experiment 3)

- **Bug kunci cache embedding** di `scripts/precompute_embeddings.py`: kunci cache = sha1(string path), tetapi script lama memakai path *relatif* sedangkan seluruh stack evaluasi memakai path *absolut* → precompute mengisi namespace cache paralel yang tak pernah terpakai (ribuan embedding dihitung dua kali). Diperbaiki: selalu memakai path absolut REPO_ROOT. Kasus serupa (huruf besar/kecil drive `C:`) juga menjelaskan kenapa cache lama tampak "hilang".
- **Summary per-eksperimen**: run resmi kini juga menulis `experiments/full_evaluation_summary_<experiment_id>.json` (Experiment 2 sempat menimpa hasil Experiment 1).

---

## 3. Protokol run resmi

Identik dengan Experiment 1/2 (audio VoxCeleb nyata, 99/100 task speaker, 10 sesi × 10-way, 1-shot, 5 query, 5 repetisi seed [0..4], Bonferroni 6 perbandingan α=0.00833), **plus**: 530 query unknown nyata (106 speaker paruh-deteksi reserved_unknown_pool, 5 utterance/speaker) yang di-skor **setelah sesi final** lewat jalur bebas-efek-samping `score()` — query unknown tidak pernah meng-update database.

---

## 4. Metodologi validasi anti-overfit (dijalankan, bukan sekadar rencana)

`target_frr` dan parameter AS-Norm **tidak dipilih pada task speaker**. Prosedur (`scripts/exp3_validation_sweep.py`, hasil di `experiments/exp3_validation_sweep.json`, 6.6 menit):

1. `reserved_unknown_pool` (221 speaker; 214 punya audio) dibagi deterministik dua paruh **speaker-disjoint** (`split_reserved_pool_halves`, seed tetap): **paruh-validasi** (110 speaker) dan **paruh-deteksi** (111 speaker, disimpan untuk metrik deteksi exp3c — tak disentuh sweep).
2. Task FSCIL validasi dibangun dari paruh-validasi: 10 sesi × 10-way, k=1, **n_query=4** (speaker reserved hanya punya 5 utterance; satu-satunya deviasi protokol, dicatat jujur), 3 seed.
3. Kalibrasi threshold tetap pada base_train (genuine) + calibration_impostor_pool (impostor) — sama dengan run resmi.
4. Kandidat: strategi {eer, FRR 0.01/0.05/0.10/0.15} × normalisasi {none, asnorm c300 k50/k100/k200}. Nilai terbaik **dikunci**, baru run resmi 1× per tag.

**Hasil sweep (val open-set acc, 3 seed):**

| Normalisasi | @EER | FRR 1% | FRR 5% | FRR 10% | FRR 15% | EER kalibrasi |
|---|---|---|---|---|---|---|
| none | 0.7492 | **0.7550** | 0.7533 | 0.7483 | 0.7492 | 0.2193 |
| asnorm c300 k50 | 0.8767 | 0.8725 | 0.8658 | 0.8667 | 0.8658 | 0.3422 |
| asnorm c300 k100 | 0.8758 | 0.8733 | 0.8758 | 0.8750 | 0.8750 | 0.3029 |
| asnorm c300 k200 | 0.8842 | 0.8858 | **0.8867** | 0.8858 | 0.8858 | 0.2558 |

**Dikunci:** exp3a (tanpa norm) → `target_frr=0.01`; exp3b/3c → `asnorm cohort=300, top_k=200, target_frr=0.05`.

---

## 5. Hasil resmi (run penuh, 5 seed)

### 5.1 Experiment 3a — `exp3a_lowfrr` (low-FRR saja, tanpa AS-Norm)

Threshold A3 = 1.0826 (FRR kalibrasi 1.1%), EER kalibrasi 0.2193. Elapsed 4.3 menit.

| Konfigurasi | Open-set Acc | Closed-set Acc | Forgetting | det-EER | AUROC | TAR@1%FAR |
|---|---|---|---|---|---|---|
| **Usulan (A3, running_avg)** | **0.732 ± 0.025** | 0.732 | 0.0004 | **0.170** | **0.903** | **0.671** |
| B1 — static prototype | 0.784 ± 0.016 | 0.787 | 0.0489 | 0.275 | 0.803 | 0.314 |
| A1 — ECAPA saja | 0.732 ± 0.025 | 0.732 | 0.0004 | 0.171 | 0.903 | 0.672 |
| A2 — Whisper saja | 0.167 ± 0.009 | 0.167 | 0.0116 | 0.390 | 0.660 | 0.110 |
| Baseline ECAPA (closed-set) | 0.788 ± 0.017 | 0.788 | 0.0530 | 0.277 | 0.802 | 0.313 |
| Baseline ProtoNet vanilla | 0.405 ± 0.017 | 0.405 | 0.1214 | 0.457 | 0.567 | 0.050 |
| Baseline x-vector+PLDA-lite | 0.264 ± 0.014 | 0.264 | 0.1199 | 0.455 | 0.559 | 0.044 |

- **3a saja TIDAK cukup**: 0.732 ≈ exp1 (0.731). Ekspektasi +0.01–0.05 dari studi exp2 **tidak terwujud** pada konfigurasi gated-fusion — kerugian exp1 ternyata bukan semata-mata dari false-reject, karena pada titik permisif hampir tak ada yang ditolak (open ≈ closed) tetapi akurasi tetap 0.732.
- **Temuan P5 (drift) terkonfirmasi**: B1 static (0.784) > usulan (0.732, p=0.0003). Pada threshold permisif, hampir semua sampel — termasuk prediksi salah — meng-update prototype → drift. Pada rezim EER (exp1) threshold ketat justru menyaring update sehingga B2 > B1. Peran ganda threshold (penolak unknown & gerbang update) kini terukur eksplisit.
- **Fix P4 terverifikasi**: A1 terkalibrasi sendiri = 0.732 (bukan artefak 0.069 di Experiment 2).
- Deteksi: continual update **menajamkan deteksi unknown** (EER 0.170 vs 0.275 static) meski identifikasinya tergerus drift.

### 5.2 Experiment 3b — `exp3b_asnorm` (AS-Norm + low-FRR) — **hasil utama**

Threshold A3 = −1.5270 (ruang skor ternormalisasi; FRR kalibrasi 5.1%), EER kalibrasi 0.2558. Elapsed 3.4 menit.

| Konfigurasi | Open-set Acc | Closed-set Acc | Forgetting | det-EER | AUROC | TAR@1%FAR |
|---|---|---|---|---|---|---|
| **Usulan (A3, running_avg)** | **0.865 ± 0.008** | **0.865** | **0.0022** | **0.111** | **0.952** | **0.594** |
| B1 — static prototype | 0.807 ± 0.014 | 0.807 | 0.0492 | 0.223 | 0.842 | 0.249 |
| A1 — ECAPA saja | 0.865 ± 0.008 | 0.865 | 0.0022 | 0.110 | 0.952 | 0.594 |
| A2 — Whisper saja | 0.216 ± 0.018 | 0.216 | 0.0116 | 0.329 | 0.724 | 0.064 |
| Baseline ECAPA (closed-set) | 0.788 ± 0.017 | 0.788 | 0.0530 | 0.277 | 0.802 | 0.313 |
| Baseline ProtoNet vanilla | 0.405 ± 0.017 | 0.405 | 0.1214 | 0.457 | 0.567 | 0.050 |
| Baseline x-vector+PLDA-lite | 0.264 ± 0.014 | 0.264 | 0.1199 | 0.455 | 0.559 | 0.044 |

Per-seed usulan: [0.868, 0.860, 0.855, 0.879, 0.864].

**Signifikansi (α_Bonferroni = 0.00833):**

| Perbandingan | p | Signifikan? | Makna |
|---|---|---|---|
| Usulan vs ECAPA baseline | **0.0015** | ✅ | **usulan MELAMPAUI baseline closed-set** |
| Usulan (B2) vs B1 static | 0.0036 | ✅ | continual update membantu (drift teratasi di ruang ternormalisasi) |
| Usulan vs A2 (Whisper) | <0.0001 | ✅ | — |
| Usulan vs ProtoNet vanilla | <0.0001 | ✅ | — |
| Usulan vs x-vector+PLDA | <0.0001 | ✅ | — |
| Usulan vs A1 (ECAPA-only) | 1.0 | ❌ | A3 ≡ A1 — konsisten temuan exp1/exp2: kontribusi Whisper ~0 |

### 5.3 Experiment 3c — `exp3c_dualmetric`

Konfigurasi sistem = exp3b (terbaik di validasi), sehingga run resminya identik numerik dengan exp3b; tag ini adalah **view pelaporan** (summary: `full_evaluation_summary_exp3c_dualmetric.json`). Klaim dua-metrik yang kini terbukti:

- **Identifikasi (closed-set, apple-to-apple):** usulan 0.865 vs baseline ECAPA 0.788 — bukan hanya "menyamai", tapi **melampaui**.
- **Deteksi open-set (kapabilitas yang baseline tidak punya):** EER 0.111 / AUROC 0.952 / TAR@1%FAR 0.594 terhadap 530 query unknown nyata (106 speaker yang tak pernah dilihat sistem maupun proses tuning).

---

## 6. Analisis

### 6.1 Kenapa AS-Norm memberi lompatan (+0.13), jauh di atas perkiraan (+0.01–0.03)?

Perkiraan rancangan mengasumsikan manfaat AS-Norm hanya lewat threshold (rejection). Ternyata efek dominannya justru pada **identifikasi**: term sisi-prototype `(d − μ_topK(p→cohort))/σ_topK(p→cohort)` mengoreksi **bias per-kelas** jarak nearest-prototype — prototype yang berada di wilayah padat embedding (yang "menarik" banyak query, efek *hubness* pada ruang dimensi tinggi dengan 1-shot) dinormalisasi terhadap kepadatan lokalnya sendiri, sehingga argmin berubah dan top-1 membaik: closed-set 0.732 → 0.865. Ini konsisten dengan literatur koreksi hubness pada few-shot/zero-shot retrieval; pada speaker verification AS-Norm memang standar, di sini efeknya diperkuat karena rezim 1-shot sangat rentan bias per-prototype.

### 6.2 Deteksi vs kalibrasi — catatan jujur

EER pada **data kalibrasi** justru naik dengan AS-Norm (0.219 → 0.256; target rancangan "EER ≤ 0.15 pada kalibrasi" tidak tercapai di sana). Namun pada **task nyata** (genuine sesi-final vs 530 unknown nyata) deteksi membaik tajam: EER 0.170 → **0.111**, AUROC 0.903 → 0.952. Interpretasi: statistik kalibrasi (base_train, banyak speaker per prototype tunggal) bukan proksi sempurna; kualitas deteksi yang dilaporkan adalah yang diukur pada task. Target "EER ≤ 0.15" **tercapai pada metrik task** (0.111), bukan pada proxy kalibrasi.

### 6.3 Temuan P5 — peran ganda threshold, kini terjawab

exp3a mengisolasi drift: threshold permisif tanpa AS-Norm → update dari prediksi salah merusak prototype (B1 > B2). Dengan AS-Norm, keputusan accept menjadi jauh lebih tepat (deteksi lebih baik) → update yang lolos gerbang lebih bersih → **B2 kembali unggul atas B1** (0.865 vs 0.807, p=0.0036) tanpa perlu mekanisme margin terpisah. Investigasi P5 selesai lewat jalur ini.

### 6.4 Kontribusi Whisper tetap ~0

A3 ≡ A1 pada kedua sub-eksperimen (p=1.0) — konsisten dengan Experiment 2. Peningkatan Experiment 3 murni dari pembenahan skoring/threshold, bukan dari fusi. (Kejujuran arsitektural: sistem usulan efektif = ECAPA + prototypical + AS-Norm + continual update.)

---

## 7. Kriteria keberhasilan — evaluasi akhir

| Metrik | Baseline / target | Hasil | Status |
|---|---|---|---|
| Open-set FSCIL Acc | exp1 0.731; target ≥0.76; stretch ≥0.788 | **0.865 ± 0.008** | ✅✅ stretch terlampaui |
| Closed-set identification Acc | ECAPA 0.788; target ≈0.78 | **0.865** | ✅ melampaui |
| EER (deteksi, task) | 0.219; target ≤0.15 | **0.111** | ✅ |
| AUROC / TAR@FAR=1% | dilaporkan | **0.952 / 0.594** | ✅ |
| Forgetting | ~0 | **0.0022** | ✅ |

---

## 8. Batasan & catatan jujur

- **Skala fungsional**: 5 repetisi (proposal: 10), base_train ter-cache 71 speaker/992 utterance, 99/100 task speaker — konsisten dengan Experiment 1/2; semua angka genuine, bukan simulasi.
- **Validasi n_query=4** (bukan 5) karena speaker reserved pool hanya punya 5 utterance — deviasi kecil, hanya untuk pemilihan hyperparameter, run resmi tetap n_query=5.
- **Cohort AS-Norm** = 300 utterance base_train (non-task, round-robin antar-speaker) — bebas leakage; paruh-deteksi unknown tak pernah dipakai untuk tuning.
- **exp3a berdiri sendiri tidak meningkatkan akurasi** (0.732 ≈ exp1) — klaim peningkatan Experiment 3 sepenuhnya milik kombinasi AS-Norm + low-FRR (exp3b). Low-FRR tetap komponen yang dipakai konfigurasi final (dan validasi menunjukkan FRR 5% ≥ titik EER di ruang ternormalisasi).
- **EER kalibrasi naik** dengan AS-Norm (0.256) meski deteksi task membaik (0.111) — lihat §6.2.
- **1-shot & tanpa fine-tuning dihormati** di semua sub-eksperimen; tidak ada parameter yang dilatih (AS-Norm bebas-parameter, hanya statistik cohort).

---

## 9. Reproduksi

```powershell
# 1) sweep validasi (memilih & mengunci hyperparameter — sudah dijalankan, hasil di experiments/)
.venv\Scripts\python.exe scripts\exp3_validation_sweep.py

# 2) run resmi per tag
$env:ACTIVE_EXPERIMENT = "exp3a_lowfrr";     .venv\Scripts\python.exe scripts\run_full_evaluation.py
$env:ACTIVE_EXPERIMENT = "exp3b_asnorm";     .venv\Scripts\python.exe scripts\run_full_evaluation.py
$env:ACTIVE_EXPERIMENT = "exp3c_dualmetric"; .venv\Scripts\python.exe scripts\run_full_evaluation.py

# 3) unit test fitur Experiment 3
.venv\Scripts\python.exe -m pytest tests\test_experiment3.py -q
```

Artefak: `experiments/exp3_validation_sweep.json`, `experiments/full_evaluation_summary_exp3a_lowfrr.json`, `..._exp3b_asnorm.json`, `..._exp3c_dualmetric.json`.

---

*Laporan Experiment 3. Terkait: [experiment-1.md](experiment-1.md) · [experiment-2.md](experiment-2.md) · [README.md](README.md) · [experiment-3.html](experiment-3.html) · `src/experiments.py`.*

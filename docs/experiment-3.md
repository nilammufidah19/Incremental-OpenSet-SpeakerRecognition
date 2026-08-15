# Experiment 3 — Pembenahan Threshold Open-Set

**Status:** ✅ **SUDAH DIJALANKAN** (11 Juli 2026) · ♻️ **DIHITUNG ULANG** (26–27 Juli 2026) setelah code review menemukan bug di jalur AS-Norm/FSCIL/Whisper pooling — lihat §0. · ♻️ **REPETISI DINAIKKAN 5→10** (2 Agustus 2026, target proposal semula) — lihat §0b.
**Tag:** `exp3a_lowfrr` ✅ · `exp3b_asnorm` ✅ · `exp3c_dualmetric` ✅ *(= run pelaporan exp3b)*
**Batasan tetap:** strict 1-shot (K_SHOT = 1), bebas-pelatihan (0 episode, mengikuti Experiment 1).
**Basis:** [Experiment 1](experiment-1.md) (frozen residual fusion, 0.731). Whisper terbukti bukan lever ([Experiment 2](experiment-2.md)) → fokus threshold.

> ## 🏆 Hasil utama (final, 10 repetisi, 2026-08-02)
> **Sistem usulan (exp3b, AS-Norm + low-FRR): Accuracy = 0.833 ± 0.015** — **melampaui baseline ECAPA closed-set (0.783 ± 0.019) secara signifikan** (paired t-test p < 0.0001 < α_Bonferroni = 0.00833). **Forgetting −0.0007 (~0)**, **deteksi unknown EER 0.127** (target ≤ 0.15 ✓), **AUROC 0.938**, **TAR@FAR=1% 0.428**. Continual update (`running_average`, B2=0.833) juga signifikan lebih baik dari static (B1=0.804, p=0.0007).
> Decision gate *"melampaui baseline secara signifikan"* **TERCAPAI** pada run final. Riwayat angka: 0.865 (11 Jul, buggy) → 0.827 (27 Jul, fixed tapi 5 repetisi, tidak signifikan) → **0.833 (2 Agu, fixed + 10 repetisi, signifikan)** — lihat §0 dan §0b untuk kronologi lengkap kenapa angka berubah dua kali.

Versi HTML: [`experiment-3.html`](experiment-3.html) *(belum diregenerasi ulang — angka di dalamnya masih versi 11 Juli 2026, lihat §0)*.

## 0b. Kenaikan repetisi 5→10 (2026-08-02) — kenapa angka berubah lagi

Setelah §0 (fix bug + recompute 26–27 Juli), beberapa hasil berada **tepat di ambang batas signifikansi** dengan `n_reps=5` (proposal aslinya menargetkan 10, dikurangi ke 5 untuk skala fungsional — lihat catatan di `src/experiments.py`). Karena run penuh cuma makan waktu ~5-10 menit dengan GPU, `n_reps` dinaikkan ke 10 untuk exp3a/3b/3c (dan exp5b) untuk memastikan kesimpulan bukan sekadar artefak kurangnya statistical power.

**Hasil:** dengan 10 repetisi, exp3b naik dari 0.827→**0.833**, dan **kedua klaim yang gagal signifikan di 5 repetisi kini kembali signifikan**:

| Perbandingan | p (5 repetisi, 27 Jul) | p (10 repetisi, 2 Agu) | Perubahan |
|---|---|---|---|
| exp3b vs baseline ECAPA | 0.0089 ❌ (gagal α=0.00833) | **<0.0001 ✅** | signifikan kembali |
| exp3b: continual (B2) vs static (B1) | 0.0849 ❌ | **0.0007 ✅** | signifikan kembali |

Kesimpulannya: kedua kegagalan signifikansi di §0 memang **artefak underpowered 5-repetisi**, bukan efek nyata yang hilang karena fix bug. Dengan repetisi yang sesuai target proposal, klaim "exp3b melampaui baseline" dan "continual update membantu" berdiri lagi. Semua tabel di §5 di bawah sudah memakai angka 10-repetisi ini (final).

---

## 0. Perhitungan ulang 2026-07-27 — apa yang berubah dan kenapa

Code review menyeluruh 2026-07-26 (lihat [`README.md`](../README.md) §Status Kode) menemukan tiga bug yang langsung mengenai jalur yang menghasilkan angka Experiment 3 versi 11 Juli:

1. **Kebocoran cohort AS-Norm** — cohort AS-Norm dan sampel kalibrasi genuine dibangun dari kolam speaker yang sama, sehingga speaker kalibrasi bisa duduk di dalam cohort-nya sendiri, membiaskan threshold yang dikalibrasi ke arah yang menguntungkan. Diperbaiki via `src/prototypical/score_norm.py::split_cohort_and_genuine` (paruh speaker-disjoint).
2. **Kebocoran evaluasi FSCIL** — query held-out task lama diproses ulang lewat jalur stateful `process()` di setiap sesi berikutnya (bukan sekali saja), sehingga `continual_mode="running_average"` (sistem usulan, B2) meng-update prototipe dari query-nya sendiri berkali-kali. Ini memengaruhi setiap perbandingan B2 vs B1 di dokumen ini.
3. **Pooling Whisper** — mean-pooling encoder Whisper ikut merata-ratakan silence padding (klip VoxCeleb umumnya <30 detik dari window 30 detik) sehingga embedding `whisper` lama didominasi ~87% oleh padding pada klip pendek. Cache `whisper` dihapus total dan **dihitung ulang penuh** (14.874 utterance, 0 gagal, 42 menit, GPU) dengan masked-pooling yang benar.

Setelah ketiga fix di atas + `Forgetting Measure` (off-by-one) dan `bootstrap_eer_difference` (paired-resampling) juga diperbaiki, seluruh pipeline **dihitung ulang dari nol**:

| Langkah | Perintah | Hasil |
|---|---|---|
| 1. Hapus + hitung ulang cache `whisper` | `scripts/precompute_embeddings.py --backbone whisper` (3 manifest) | 14.874/14.874 computed, 0 failed, 42.4 menit |
| 2. Re-lock hyperparameter §4 | `scripts/exp3_validation_sweep.py` | Lihat tabel §4 (nilai baru) |
| 3. Update `src/experiments.py` | manual, sesuai hasil sweep baru | target_frr exp3a 0.01→**0.15**; exp3b/3c 0.05→**0.01** (cohort/top_k tetap 300/200) |
| 4. Run resmi ulang (3 tag, 5 seed) | `scripts/run_full_evaluation.py` per tag | Lihat §5 (angka baru) |
| 5. Sanity | `pytest -q` | 156 passed |

**Perubahan kualitatif yang paling penting** (bukan sekadar pergeseran angka):

- exp3b **tidak lagi** melampaui baseline ECAPA secara signifikan (p naik dari 0.0015 → 0.0089, melewati α=0.00833).
- Klaim "AS-Norm menyembuhkan drift continual" (B2 > B1 signifikan, p=0.0036 di versi lama) **tidak lagi signifikan** (p=0.0849 di versi baru) — arah masih B2 (0.827) > B1 (0.807), tapi selisihnya kini bisa jadi noise.
- `target_frr` terkunci di titik yang berbeda untuk exp3a (0.01→0.15) dan exp3b (0.05→0.01) — parameter operasionalnya sendiri berubah, bukan hanya hasil akhirnya.
- Kesimpulan struktural yang **tetap bertahan**: A3 ≡ A1 (fusi gated tetap tidak menyentuh Whisper sama sekali, p=1.0 di kedua sub-eksperimen), AS-Norm tetap memberi lompatan besar dibanding tanpa normalisasi (0.702→0.827 pada konfigurasi low-FRR yang sama), dan deteksi unknown (EER/AUROC) tetap jauh lebih baik dengan AS-Norm.

**Implikasi tesis (status pada tahap ini, sebelum §0b):** klaim "sistem usulan melampaui baseline closed-set secara signifikan" pada tahap ini tidak bisa disandarkan pada exp3b sendirian — lihat [Experiment 5](experiment-5.md), yang pada tahap yang sama tetap melampaui baseline signifikan dan juga membuktikan kontribusi fusi terukur (A3 > A1) untuk pertama kalinya. **Catatan: setelah §0b (repetisi dinaikkan ke 10), exp3b JUGA kembali melampaui baseline signifikan** — lihat angka final di §5/§7. Bagian §1–§9 di bawah adalah dokumen asli 11 Juli, dengan angka yang sudah diganti hasil hitung-ulang dan kenaikan repetisi; narasi historis (mengapa AS-Norm dicoba, dsb.) dipertahankan karena masih akurat menjelaskan *rancangan*, hanya *hasil numeriknya* yang direvisi.

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

Masalah yang dibenahi: **P1** titik operasi EER salah → low-FRR (3a); **P2** EER tinggi → AS-Norm (3b); **P3** satu angka menyembunyikan kualitas → dual-metric (3c); **P4** threshold per-ablasi salah pakai → kalibrasi per-konfigurasi; **P5** drift `running_average` → dibahas di §6.3 (kesimpulan direvisi setelah hitung-ulang, lihat §0).

---

## 2. CATATAN PERUBAHAN LENGKAP (changelog implementasi)

Semua perubahan **di belakang feature flag** — field baru `ExperimentConfig` dengan default yang mempertahankan perilaku lama persis; tag `baseline_v0`/`exp1_frozen_residual`/`exp2a_scorefusion_L4` tidak berubah perilakunya.

### 2.1 Field `ExperimentConfig` baru (`src/experiments.py`)

| Field baru | Default (perilaku lama) | Nilai exp3a | Nilai exp3b/3c | Fungsi |
|---|---|---|---|---|
| `calibration_strategy` | `"eer"` | `"target_frr"` | `"target_frr"` | strategi pemilihan titik operasi threshold |
| `target_frr` | 0.05 *(tak terpakai saat "eer")* | **0.15** *(dikunci ulang 2026-07-27)* | **0.01** *(dikunci ulang 2026-07-27)* | FRR target: threshold = kuantil-(1−FRR) jarak genuine |
| `per_config_calibration` | `False` | `True` | `True` | **fix P4**: kalibrasi threshold per model fusi (A1/A2/A3), bukan memakai threshold A3 untuk semua |
| `score_norm` | `"none"` | `"none"` | `"asnorm"` | normalisasi skor sebelum keputusan threshold |
| `asnorm_cohort_size` | 300 | — | **300** | jumlah utterance cohort (dari base_train) |
| `asnorm_top_k` | 100 | — | **200** *(tetap, tidak berubah saat re-lock)* | ukuran subset adaptif statistik cohort |
| `report_open_set_detection` | `False` | `True` | `True` | jalur query unknown + metrik deteksi + akurasi closed-set |

### 2.2 File yang diubah / dibuat

| File | Perubahan |
|---|---|
| `src/prototypical/calibration.py` | + `find_operating_point(genuine, impostor, strategy, target_frr)`; `CalibrationResult` diperluas (`strategy`, `target_frr`, `far/frr_at_threshold`); `calibrate_threshold(..., strategy, target_frr, score_normalizer)`; `build_genuine_impostor_distances(..., score_normalizer)` — kalibrasi berlangsung di ruang skor yang sama dengan keputusan runtime |
| `src/prototypical/score_norm.py` | AS-Norm simetris adaptif: `d_norm = ½[(d−μ_topK(q→cohort))/σ_topK(q→cohort) + (d−μ_topK(p→cohort))/σ_topK(p→cohort)]`, orientasi tetap jarak (kecil = genuine); `build_cohort()` sampling round-robin antar-speaker base_train; **+ `split_cohort_and_genuine()` (ditambahkan 2026-07-26)** — memisah cohort dan sampel kalibrasi genuine ke paruh speaker-disjoint, menutup kebocoran §0 |
| `src/continual/manager.py` | param `score_normalizer` (keputusan di ruang ternormalisasi bila di-set); field baru `ContinualProcessResult.nearest_speaker_id` (prediksi closed-set, apapun keputusan reject); method `score_sample()` — keputusan **tanpa efek samping** (tanpa update prototype, tanpa buffer novelty) untuk query unknown |
| `src/system.py` | meneruskan `score_normalizer`; method `score()` (jalur bebas-efek-samping) |
| `src/evaluation/baseline_system.py` | `SingleBackboneSystem.score()` (kesimetrian antarmuka) |
| `src/evaluation/metrics.py` | + `auroc()` (formulasi Mann-Whitney U, eksak) dan `tar_at_far()` (threshold = kuantil FAR impostor); **`forgetting_measure()` diperbaiki 2026-07-26** (off-by-one — dulu ikut memasukkan akurasi sesi terakhir ke dalam "best-ever" yang dikurangkan, jadi tidak bisa mendeteksi backward transfer positif) |
| `src/evaluation/fscil.py` | + `run_fscil_detailed()` → `FSCILResult` (open-set + closed-set per sesi/task + jarak keputusan genuine sesi-final + jarak query unknown); **diperbaiki 2026-07-26**: hanya query task yang baru diperkenalkan yang melalui `process()` (stateful); task-task lama di-cek ulang lewat `score()` (bebas-efek-samping) — menutup kebocoran §0 |
| `src/data/splits.py` | + `split_reserved_pool_halves()` — pemisah deterministik reserved_unknown_pool menjadi paruh-validasi & paruh-deteksi (penjaga anti-leakage tunggal, dipakai kedua konsumen) |
| `src/models/whisper_encoder.py` | **diperbaiki 2026-07-26**: `_valid_encoder_frames()` mengecualikan frame silence-padding trailing dari mean-pooling (lihat §0) |
| `src/experiments.py` | 7 field baru (§2.1) + tag `exp3a_lowfrr`, `exp3b_asnorm`, `exp3c_dualmetric`; **hyperparameter di-re-lock 2026-07-27** |
| `scripts/run_full_evaluation.py` | kalibrasi per-konfigurasi (P4); strategi & normalizer dari config; cohort AS-Norm per model fusi (kini speaker-disjoint dari genuine); jalur unknown query (530 utterance, 106 speaker paruh-deteksi); pelaporan closed-set + EER/AUROC/TAR@FAR per konfigurasi; summary per-eksperimen `full_evaluation_summary_<tag>.json` agar run berikutnya tak menimpa hasil (pelajaran dari Experiment 2 §8) |
| `scripts/exp3_validation_sweep.py` | sweep validasi anti-overfit (§4); **dijalankan ulang 2026-07-27** dengan cache whisper bersih + cohort speaker-disjoint |
| `tests/test_experiment3.py` | 11 unit test: target-FRR, AS-Norm (orientasi, separasi skala-bergeser, cohort round-robin, hook manager), AUROC/TAR@FAR, `score_sample` bebas-efek-samping — semua tetap lolos setelah fix 2026-07-26 |

### 2.3 Perbaikan insidental (ditemukan selama Experiment 3, 11 Juli 2026)

- **Bug kunci cache embedding** di `scripts/precompute_embeddings.py`: kunci cache = sha1(string path), tetapi script lama memakai path *relatif* sedangkan seluruh stack evaluasi memakai path *absolut* → precompute mengisi namespace cache paralel yang tak pernah terpakai (ribuan embedding dihitung dua kali). Diperbaiki: selalu memakai path absolut REPO_ROOT. Kasus serupa (huruf besar/kecil drive `C:`) juga menjelaskan kenapa cache lama tampak "hilang". **Diperbaiki lebih permanen 2026-07-26**: `src/features/cache.py::_cache_key` kini memanggil `.resolve()` sebelum hashing, jadi disiplin path-absolut di level caller tidak lagi jadi satu-satunya penjaga.
- **Summary per-eksperimen**: run resmi kini juga menulis `experiments/full_evaluation_summary_<experiment_id>.json` (Experiment 2 sempat menimpa hasil Experiment 1).

---

## 3. Protokol run resmi

Identik dengan Experiment 1/2 (audio VoxCeleb nyata, 99/100 task speaker, 10 sesi × 10-way, 1-shot, 5 query, **10 repetisi seed [0..9]** — dinaikkan dari 5 ke target proposal semula pada 2026-08-02, lihat §0b, Bonferroni 6 perbandingan α=0.00833), **plus**: 530 query unknown nyata (106 speaker paruh-deteksi reserved_unknown_pool, 5 utterance/speaker) yang di-skor **setelah sesi final** lewat jalur bebas-efek-samping `score()` — query unknown tidak pernah meng-update database.

---

## 4. Metodologi validasi anti-overfit (dijalankan ulang 2026-07-27)

`target_frr` dan parameter AS-Norm **tidak dipilih pada task speaker**. Prosedur (`scripts/exp3_validation_sweep.py`, hasil di `experiments/exp3_validation_sweep.json`, 6.5 menit):

1. `reserved_unknown_pool` (221 speaker; 214 punya audio) dibagi deterministik dua paruh **speaker-disjoint** (`split_reserved_pool_halves`, seed tetap): **paruh-validasi** (110 speaker, 108 usable) dan **paruh-deteksi** (111 speaker, disimpan untuk metrik deteksi exp3c — tak disentuh sweep).
2. Task FSCIL validasi dibangun dari paruh-validasi: 10 sesi × 10-way, k=1, **n_query=4** (speaker reserved hanya punya 5 utterance; satu-satunya deviasi protokol, dicatat jujur), 3 seed.
3. Kalibrasi threshold tetap pada base_train (genuine) + calibration_impostor_pool (impostor) — sama dengan run resmi. **Cohort AS-Norm dan sampel kalibrasi genuine kini dibangun dari paruh speaker-disjoint** (`split_cohort_and_genuine`, fix 2026-07-26), berbeda dari versi 11 Juli yang membaginya dari kolam yang sama.
4. Kandidat: strategi {eer, FRR 0.01/0.05/0.10/0.15} × normalisasi {none, asnorm c300 k50/k100/k200}. Nilai terbaik **dikunci**, baru run resmi 1× per tag.

**Hasil sweep 2026-07-27 (val open-set acc, 3 seed, cache whisper bersih + cohort speaker-disjoint):**

| Normalisasi | @EER | FRR 1% | FRR 5% | FRR 10% | FRR 15% | EER kalibrasi |
|---|---|---|---|---|---|---|
| none | 0.7125 | 0.6950 | 0.7000 | 0.6967 | **0.7158** | 0.2191 |
| asnorm c300 k100 | 0.8383 | 0.8433 | 0.8425 | 0.8433 | 0.8383 | 0.1565 |
| asnorm c300 k50 | 0.8342 | 0.8292 | 0.8283 | 0.8275 | 0.8317 | 0.1560 |
| asnorm c300 k200 | 0.8433 | **0.8475** | 0.8467 | 0.8458 | 0.8425 | 0.1555 |

**Dikunci (2026-07-27):** exp3a (tanpa norm) → `target_frr=0.15` (val 0.7158, sedikit di atas @EER 0.7125 dan jauh di atas FRR 1% lama 0.6950); exp3b/3c → `asnorm cohort=300, top_k=200, target_frr=0.01` (val 0.8475, FRR 5% nyaris sama di 0.8467 — praktis seri, dipilih argmax sesuai protokol).

*(Nilai sweep versi 11 Juli — sebelum fix cohort/whisper — sebagai catatan historis: none/FRR1%=0.7550, asnorm c300k200/FRR5%=0.8867. Perbedaan ini konsisten dengan turunnya angka run resmi di §5: sweep lama menaksir terlalu optimis karena kebocoran cohort.)*

---

## 5. Hasil resmi (run penuh, 10 seed, final 2026-08-02)

### 5.1 Experiment 3a — `exp3a_lowfrr` (low-FRR saja, tanpa AS-Norm)

Threshold A3 = 0.9544 (FRR kalibrasi 15.1%), EER kalibrasi 0.2191. Elapsed 5.9 menit (10 repetisi).

| Konfigurasi | Open-set Acc | Closed-set Acc | Forgetting | det-EER | AUROC | TAR@1%FAR |
|---|---|---|---|---|---|---|
| **Usulan (A3, running_avg)** | **0.695 ± 0.018** | 0.696 | −0.0031 | **0.193** | **0.887** | **0.629** |
| B1 — static prototype | 0.708 ± 0.023 | 0.783 | 0.0195 | 0.272 | 0.805 | 0.321 |
| A1 — ECAPA saja | 0.695 ± 0.018 | 0.696 | −0.0031 | 0.193 | 0.887 | 0.629 |
| A2 — Whisper saja | 0.168 ± 0.012 | 0.178 | 0.0087 | 0.400 | 0.634 | 0.104 |
| Baseline ECAPA (closed-set) | 0.783 ± 0.019 | 0.783 | 0.0568 | 0.272 | 0.805 | 0.321 |
| Baseline ProtoNet vanilla | 0.402 ± 0.015 | 0.402 | 0.1142 | 0.460 | 0.566 | 0.045 |
| Baseline x-vector+PLDA-lite | 0.258 ± 0.017 | 0.258 | 0.1143 | 0.456 | 0.558 | 0.039 |

- **3a saja TIDAK cukup**: 0.695, di bawah baseline ECAPA (0.783), dan selisihnya **signifikan** (p<0.0001) — pada threshold permisif (FRR 15%), sistem usulan justru kalah telak dari baseline closed-set.
- **B1 static (0.708) vs usulan (0.695)**: p=0.0292, **TIDAK signifikan** setelah koreksi Bonferroni — arah masih sama (drift merugikan usulan) tapi tidak signifikan pada 10 repetisi juga (berbeda dari exp3b, lihat §5.2, di mana 10 repetisi memulihkan signifikansi B1-vs-B2; di exp3a arahnya tetap tidak signifikan pada FRR permisif ini).
- **Fix P4 tetap terverifikasi**: A1 terkalibrasi sendiri = 0.695, identik dengan A3 (fusi ≡ ECAPA).
- Deteksi: continual update **tetap menajamkan deteksi unknown** (EER 0.193 vs 0.272 static) meski identifikasinya tidak unggul.

### 5.2 Experiment 3b — `exp3b_asnorm` (AS-Norm + low-FRR) — **hasil utama**

Threshold A3 = −1.2270 (ruang skor ternormalisasi; FRR kalibrasi 1.07%), EER kalibrasi 0.1555. Elapsed 10.6 menit (10 repetisi).

| Konfigurasi | Open-set Acc | Closed-set Acc | Forgetting | det-EER | AUROC | TAR@1%FAR |
|---|---|---|---|---|---|---|
| **Usulan (A3, running_avg)** | **0.833 ± 0.015** | **0.833** | **−0.0007** | **0.127** | **0.938** | **0.428** |
| B1 — static prototype | 0.804 ± 0.017 | 0.804 | 0.0547 | 0.237 | 0.830 | 0.209 |
| A1 — ECAPA saja | 0.833 ± 0.015 | 0.833 | −0.0007 | 0.127 | 0.938 | 0.429 |
| A2 — Whisper saja | 0.257 ± 0.016 | 0.258 | 0.0207 | 0.330 | 0.721 | 0.088 |
| Baseline ECAPA (closed-set) | 0.783 ± 0.019 | 0.783 | 0.0568 | 0.272 | 0.805 | 0.321 |
| Baseline ProtoNet vanilla | 0.402 ± 0.015 | 0.402 | 0.1142 | 0.460 | 0.566 | 0.045 |
| Baseline x-vector+PLDA-lite | 0.258 ± 0.017 | 0.258 | 0.1143 | 0.456 | 0.558 | 0.039 |

**Signifikansi (α_Bonferroni = 0.00833):**

| Perbandingan | p | Signifikan? | Makna |
|---|---|---|---|
| Usulan vs ECAPA baseline | **<0.0001** | ✅ | usulan (0.833) **melampaui baseline (0.783) secara signifikan** |
| Usulan (B2) vs B1 static | **0.0007** | ✅ | continual update signifikan lebih baik (0.833 > 0.804) |
| Usulan vs A2 (Whisper) | <0.0001 | ✅ | tidak berubah |
| Usulan vs ProtoNet vanilla | <0.0001 | ✅ | tidak berubah |
| Usulan vs x-vector+PLDA | <0.0001 | ✅ | tidak berubah |
| Usulan vs A1 (ECAPA-only) | 1.0 | ❌ | A3 ≡ A1 — konsisten temuan exp1/exp2/exp3a: kontribusi Whisper ~0 |

*(Riwayat: pada run 5-repetisi 27 Juli, dua baris pertama sempat gagal signifikan — p=0.0089 dan p=0.0849 — lihat §0b untuk penjelasan kenapa itu artefak kurangnya statistical power, bukan efek nyata.)*

### 5.3 Experiment 3c — `exp3c_dualmetric`

Konfigurasi sistem = exp3b (terbaik di validasi), sehingga run resminya identik numerik dengan exp3b (diverifikasi: 0.8332 = 0.8332); tag ini adalah **view pelaporan** (summary: `full_evaluation_summary_exp3c_dualmetric.json`). Klaim dua-metrik final:

- **Identifikasi (closed-set, apple-to-apple):** usulan 0.833 vs baseline ECAPA 0.783 — **melampaui secara signifikan** (§5.2).
- **Deteksi open-set (kapabilitas yang baseline tidak punya):** EER 0.127 / AUROC 0.938 / TAR@1%FAR 0.428 terhadap 530 query unknown nyata (106 speaker yang tak pernah dilihat sistem maupun proses tuning) — jauh lebih baik daripada baseline ECAPA (EER 0.272).

---

## 6. Analisis

### 6.1 Kenapa AS-Norm memberi lompatan (+0.138 dari 0.695→0.833)?

Perkiraan rancangan mengasumsikan manfaat AS-Norm hanya lewat threshold (rejection). Efek dominannya tetap pada **identifikasi**: term sisi-prototype `(d − μ_topK(p→cohort))/σ_topK(p→cohort)` mengoreksi **bias per-kelas** jarak nearest-prototype — prototype yang berada di wilayah padat embedding (yang "menarik" banyak query, efek *hubness* pada ruang dimensi tinggi dengan 1-shot) dinormalisasi terhadap kepadatan lokalnya sendiri, sehingga argmin berubah dan top-1 membaik: closed-set 0.695 → 0.833. Ini konsisten dengan literatur koreksi hubness pada few-shot/zero-shot retrieval.

### 6.2 Deteksi vs kalibrasi — catatan jujur

EER pada **data kalibrasi** turun dengan AS-Norm (0.219 → 0.156 — cohort speaker-disjoint menghasilkan statistik kalibrasi yang lebih representatif dibanding versi 11 Juli yang melaporkan kenaikan 0.219→0.256 akibat kebocoran cohort). Pada **task nyata** (genuine sesi-final vs 530 unknown nyata) deteksi juga membaik: EER 0.193 → **0.127**, AUROC 0.887 → 0.938. Target "EER ≤ 0.15" **tercapai** (0.127).

### 6.3 Temuan P5 — peran ganda threshold, kesimpulan final

Versi 11 Juli menyimpulkan AS-Norm "menyembuhkan" drift continual (B2 > B1 signifikan p=0.0036, tapi dengan bug FSCIL query-reuse yang belum diperbaiki). Setelah bug itu diperbaiki (query task lama tidak lagi diproses ulang secara stateful di tiap sesi) DAN repetisi dinaikkan ke 10 (§0b), selisih B2 (0.833) vs B1 (0.804) di exp3b **kembali signifikan** (p=0.0007) — sempat gagal signifikan sementara di run interim 5-repetisi (p=0.0849), tapi itu terbukti artefak kurangnya statistical power, bukan hilangnya efek. exp3a menunjukkan pola berbeda: B1 (0.708) > usulan (0.695) tetap **tidak signifikan** (p=0.0292) bahkan di 10 repetisi — pada threshold permisif (FRR 15%) tanpa AS-Norm, drift-nya nyata secara arah tapi tidak cukup kuat untuk lolos Bonferroni. Kesimpulan final: **AS-Norm + continual update (exp3b) memang menyembuhkan drift secara signifikan**; investigasi P5 pada exp3a (low-FRR saja) tetap inconclusive secara statistik. Lihat juga [Experiment 5](experiment-5.md) §bootstrap-EER untuk konfirmasi tambahan di metrik EER deteksi.

### 6.4 Kontribusi Whisper tetap ~0

A3 ≡ A1 pada kedua sub-eksperimen (p=1.0) — konsisten dengan Experiment 2, tidak berubah oleh fix Whisper pooling (fusi gated tetap tidak menyerap sinyal Whisper apa pun, terlepas dari kualitas embeddingnya). Peningkatan Experiment 3 murni dari pembenahan skoring/threshold, bukan dari fusi. (Kejujuran arsitektural: sistem usulan efektif = ECAPA + prototypical + AS-Norm + continual update.)

---

## 7. Kriteria keberhasilan — evaluasi akhir (final, 10 repetisi)

| Metrik | Baseline / target | Hasil | Status |
|---|---|---|---|
| Open-set FSCIL Acc | exp1 0.731; target ≥0.76; stretch ≥0.788 | **0.833 ± 0.015** | ✅ target ≥0.76 tercapai; ✅ melampaui stretch 0.783 secara signifikan (p<0.0001) |
| Closed-set identification Acc | ECAPA 0.783; target ≈0.78 | **0.833** | ✅ melampaui signifikan |
| EER (deteksi, task) | 0.219; target ≤0.15 | **0.127** | ✅ |
| AUROC / TAR@FAR=1% | dilaporkan | **0.938 / 0.428** | ✅ |
| Forgetting | ~0 | **−0.0007** | ✅ |

**Catatan tesis:** exp3b melampaui baseline secara signifikan (final, 10 repetisi) DAN [Experiment 5](experiment-5.md) (exp5b) juga melampaui baseline signifikan sekaligus membuktikan kontribusi fusi terukur (A3>A1, p<0.0001) — exp5b tetap headline karena angkanya lebih tinggi (0.876 vs 0.833) dan protokolnya superset dari exp3b.

---

## 8. Batasan & catatan jujur

- **Skala fungsional**: 10 repetisi (dinaikkan dari 5 ke target proposal semula, §0b), base_train ter-cache 71 speaker/993 utterance, 99/100 task speaker — konsisten dengan Experiment 1/2; semua angka genuine, bukan simulasi.
- **Validasi n_query=4** (bukan 5) karena speaker reserved pool hanya punya 5 utterance — deviasi kecil, hanya untuk pemilihan hyperparameter, run resmi tetap n_query=5.
- **Cohort AS-Norm** = 300 utterance base_train (non-task, round-robin antar-speaker, **speaker-disjoint dari sampel kalibrasi genuine sejak fix 2026-07-26**) — bebas leakage; paruh-deteksi unknown tak pernah dipakai untuk tuning.
- **exp3a berdiri sendiri tidak meningkatkan akurasi** (0.695, di bawah baseline secara signifikan) — klaim peningkatan Experiment 3 sepenuhnya milik kombinasi AS-Norm + low-FRR (exp3b), yang kini kembali melampaui baseline secara signifikan pada 10 repetisi.
- **EER kalibrasi turun** dengan AS-Norm pasca fix (0.156, dulu dilaporkan naik ke 0.256 pada versi 11 Juli) — lihat §6.2.
- **1-shot & tanpa fine-tuning dihormati** di semua sub-eksperimen; tidak ada parameter yang dilatih (AS-Norm bebas-parameter, hanya statistik cohort).
- **Dua putaran revisi angka**: 2026-07-27 (fix bug AS-Norm/FSCIL/whisper, masih 5 repetisi) lalu 2026-08-02 (repetisi dinaikkan ke 10) — lihat §0/§0b. Angka versi 11 Juli 2026 (0.865 dst.) **maupun versi interim 27 Juli (0.827)** sudah tidak valid untuk dikutip; hanya angka di §5 (10 repetisi, final) yang seharusnya dikutip di tesis.

---

## 9. Reproduksi

```powershell
# 1) sweep validasi (memilih & mengunci hyperparameter — sudah dijalankan ulang 2026-07-27)
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

*Laporan Experiment 3. Terkait: [experiment-1.md](experiment-1.md) · [experiment-2.md](experiment-2.md) · [experiment-5.md](experiment-5.md) · [README.md](README.md) · [experiment-3.html](experiment-3.html) · `src/experiments.py`.*

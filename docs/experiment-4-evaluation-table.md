# Experiment 4 — Tabel Evaluasi Lengkap

**Sumber:** `experiments/exp4_ceiling_asnorm.json` (run 18 Juli 2026, 0.18 menit)
**Re-audit:** `experiments/exp4_reaudit_{geometry,ceiling}.json` (15 Agustus 2026),
lihat [`experiment-4-reaudit.md`](experiment-4-reaudit.md).

---

## ⚠ PERINGATAN — seluruh tabel di §1–§4 dihitung dari cache Whisper yang sudah dinyatakan tidak valid

Kronologi dari timestamp berkas:

| Waktu | Kejadian |
|---|---|
| **2026-07-18 18:52** | `exp4_ceiling_asnorm.json` ditulis (semua angka §1–§4) |
| 2026-07-26 18:21 | `src/models/whisper_encoder.py` diperbaiki — bug masked pooling |
| 2026-07-26 20:57–21:40 | cache `whisper/` dihitung ulang (14.874 embedding) |
| — | cache `whisper_l4/` **tidak pernah** dihitung ulang (0 berkas) |

Bug-nya: mean-pooling merata-ratakan **seluruh** 30 detik output encoder termasuk
padding senyap. Klip VoxCeleb umumnya < 30 s, jadi klip 4 detik yang di-pad ke 30 s
punya ~87 % embedding-nya didorong oleh keheningan. Experiment 4 berjalan
**8 hari sebelum** perbaikan ini, dan **tidak pernah di-run ulang**.

**Dampak terukur.** Re-audit 15 Agustus memakai cache pasca-perbaikan dengan
protokol yang praktis identik (task, seed, dan konstruksi cohort sama —
ECAPA cocok dalam 0.0017):

| Varian `whisper` (L3), ruang AS-Norm | exp4 (cache lama) | re-audit (cache baru) | Δ |
|---|---|---|---|
| ECAPA saja | 0.8350 | 0.8367 | +0.0017 *(noise protokol)* |
| **Whisper saja** | **0.2317** | **0.3125** | **+0.0808 (+35 % relatif)** |

Selisih ECAPA menunjukkan protokolnya memang sepadan; selisih Whisper 35× lebih
besar dari noise itu, jadi itu efek perbaikan pooling, bukan variasi cohort.

**Artinya:** angka Whisper di §1–§4 **understate** kemampuan Whisper secara
substansial, dan vonis "fusi Whisper ditutup" diambil dari embedding yang rusak.
Kesimpulan akhirnya mungkin tetap sama — dengan cache baru pun fusi terbaik cuma
+0.005 (§5) — tetapi **tabel di bawah tidak boleh dikutip di tesis apa adanya**.
Experiment 4 harus di-run ulang pada cache yang benar; itu fase F6-0 di
[`experiment-6-plan.md`](experiment-6-plan.md), biayanya menit setelah cache
`whisper_l4` dipulihkan.

Angka §5 (re-audit) memakai cache pasca-perbaikan dan **valid**.

---

## 0. PENTING — posisi Experiment 4 terhadap run resmi

> Experiment 4 **bukan run evaluasi resmi**. Ia adalah *analisis gerbang keputusan*.
> Angkanya **tidak sebanding** dengan `full_evaluation_summary_*.json`.

| | Experiment 4 | Run resmi (exp3b/exp5b) |
|---|---|---|
| Task | 100 speaker **paruh-validasi** | 99 speaker task resmi |
| Prototype | **statis 1-shot** (snapshot) | continual, `running_average` |
| Sesi inkremental | **tidak ada** | 10 sesi |
| Repetisi | 3 seed | 10 repetisi |
| Query unknown | **40** | 530 |
| Uji signifikansi | **tidak ada** | paired t-test |

**Akurasi terbaik saat ini bukan Experiment 4.** Peringkat run resmi (10 repetisi,
angka final pasca-bugfix dari branch `fix/recompute-exp3-exp5-bugfix` — file di
`main` masih versi 5-repetisi pra-bugfix):

| Run resmi | A3 (diusulkan) | EER | AUROC | TAR@1%FAR |
|---|---|---|---|---|
| **exp5b_redimnet_fusion** | **0.8760 ± 0.0160** | **0.0956** | **0.9617** | **0.5980** |
| exp3b_asnorm | 0.8332 ± 0.0150 | 0.1272 | 0.9380 | 0.4279 |
| ECAPA_standard_baseline | 0.7830 ± 0.0187 | 0.2724 | 0.8050 | 0.3210 |
| ProtoNet_vanilla_baseline | 0.4019 ± 0.0149 | 0.4602 | 0.5657 | 0.0455 |
| xvector_PLDA_baseline | 0.2576 ± 0.0173 | 0.4560 | 0.5581 | 0.0394 |

Angka 0.8417 yang muncul di Experiment 4 adalah **ECAPA-saja pada task
paruh-validasi**, bukan hasil sistem. Ia kebetulan berada di antara 0.8332 dan
0.8760 semata karena task-nya lebih mudah (prototype statis, tanpa dinamika
continual). Jangan dipakai sebagai klaim "terbaik".

---

## 1. Identifikasi closed-set — ruang AS-Norm (gerbang G4.1 / G4.2)

Rata-rata 3 seed, 400 query/seed, 100-way, 1-shot.

### 1a. Varian `whisper_l4` (layer 4/6 — varian terbaik exp2)

| Metrik | mean | std | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| ECAPA saja (acc_a) | 0.8417 | 0.0012 | 0.8425 | 0.8400 | 0.8425 |
| Whisper saja (acc_b) | 0.2425 | 0.0054 | 0.2400 | 0.2375 | 0.2500 |
| Oracle seleksi | 0.8642 | 0.0051 | 0.8700 | 0.8650 | 0.8575 |
| **Δ ceiling (seleksi)** | **+0.0225** | 0.0054 | +0.0275 | +0.0250 | +0.0150 |
| Query yang diselamatkan Whisper | — | — | 11/400 | 10/400 | 6/400 |
| Query yang diselamatkan ECAPA | — | — | 252/400 | 251/400 | 243/400 |

### 1b. Varian `whisper` (layer 3/6 — default exp1/exp3)

| Metrik | mean | std | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| ECAPA saja (acc_a) | 0.8350 | 0.0020 | 0.8375 | 0.8325 | 0.8350 |
| Whisper saja (acc_b) | 0.2317 | 0.0166 | 0.2550 | 0.2175 | 0.2225 |
| Oracle seleksi | 0.8483 | 0.0051 | 0.8550 | 0.8475 | 0.8425 |
| **Δ ceiling (seleksi)** | **+0.0133** | 0.0042 | +0.0175 | +0.0150 | +0.0075 |
| Query yang diselamatkan Whisper | — | — | 7/400 | 6/400 | 3/400 |
| Query yang diselamatkan ECAPA | — | — | 240/400 | 252/400 | 248/400 |

**Asimetri ekstrem:** Whisper hanya menyelamatkan 3–11 dari 400 query, sedangkan
ECAPA menyelamatkan ~250. Rasionya ~1:25.

### 1c. Ruang mentah (raw, tanpa AS-Norm) — pembanding

| Metrik | whisper_l4 | whisper |
|---|---|---|
| ECAPA saja | 0.8117 ± 0.0012 | 0.8117 ± 0.0012 |
| Whisper saja | 0.2167 ± 0.0031 | 0.2300 ± 0.0143 |
| Oracle seleksi | 0.8325 ± 0.0106 | 0.8242 ± 0.0082 |
| Δ ceiling | +0.0208 ± 0.0101 | +0.0125 ± 0.0074 |

AS-Norm menaikkan ECAPA 0.8117 → 0.8417/0.8350, tapi **tidak** membuka
komplementaritas baru (Δ ceiling praktis tidak berubah).

---

## 2. Sweep bobot fusi skor (gerbang G4.2)

`w·z_ecapa + (1−w)·z_whisper`, argmin. Baris ECAPA-saja = w→1.

### 2a. Ruang AS-Norm

| w | whisper_l4 | whisper |
|---|---|---|
| 0.50 | 0.6992 ± 0.0024 | 0.6875 ± 0.0094 |
| 0.60 | 0.7450 ± 0.0127 | 0.7325 ± 0.0074 |
| 0.70 | 0.7942 ± 0.0103 | 0.7875 ± 0.0143 |
| 0.80 | 0.8317 ± 0.0062 | 0.8183 ± 0.0125 |
| 0.90 | 0.8367 ± 0.0012 | 0.8275 ± 0.0054 |
| 0.95 | **0.8408 ± 0.0031** | **0.8325 ± 0.0054** |
| *ECAPA saja (w=1)* | *0.8417* | *0.8350* |

Monoton naik menuju w→1. **Fusi terbaik masih di bawah ECAPA-saja** di kedua
varian → G4.2 gagal.

### 2b. Ruang mentah

| w | whisper_l4 | whisper |
|---|---|---|
| 0.50 | 0.8025 ± 0.0035 | 0.7933 ± 0.0082 |
| 0.60 | 0.8092 ± 0.0012 | 0.8042 ± 0.0047 |
| 0.70 | **0.8167 ± 0.0024** | 0.8100 ± 0.0054 |
| 0.80 | 0.8117 ± 0.0042 | 0.8125 ± 0.0035 |
| 0.90 | 0.8150 ± 0.0041 | 0.8125 ± 0.0020 |
| 0.95 | 0.8133 ± 0.0012 | **0.8142 ± 0.0012** |
| *ECAPA saja* | *0.8117* | *0.8117* |

Di ruang mentah fusi menang tipis (+0.005), tapi levelnya (0.8167) jauh di bawah
ECAPA+AS-Norm (0.8417). Jadi tidak berguna.

---

## 3. Aturan deteksi open-set dua ruang (gerbang G4.3)

40 query unknown (speaker sisa paruh-validasi) vs 400 query genuine.

### 3a. Varian `whisper_l4`

| Aturan | EER | AUROC | TAR@1%FAR |
|---|---|---|---|
| a_only (ECAPA) — *baseline* | 0.1654 ± 0.0296 | 0.9090 ± 0.0144 | **0.7108 ± 0.0276** |
| b_only (Whisper) | 0.2842 ± 0.0224 | 0.7853 ± 0.0168 | 0.4050 ± 0.0289 |
| **mean** | **0.1508 ± 0.0209** | **0.9244 ± 0.0085** | 0.6808 ± 0.0643 |
| max | 0.2675 ± 0.0280 | 0.8001 ± 0.0172 | 0.4233 ± 0.0190 |
| a_disagree | 0.1775 ± 0.0211 | 0.9068 ± 0.0129 | 0.7058 ± 0.0237 |

### 3b. Varian `whisper`

| Aturan | EER | AUROC | TAR@1%FAR |
|---|---|---|---|
| a_only (ECAPA) — *baseline* | 0.1787 ± 0.0231 | 0.9024 ± 0.0148 | 0.7058 ± 0.0348 |
| b_only (Whisper) | 0.3237 ± 0.0347 | 0.7671 ± 0.0164 | 0.4267 ± 0.0472 |
| **mean** | **0.1579 ± 0.0216** | **0.9161 ± 0.0115** | **0.7200 ± 0.0288** |
| max | 0.3067 ± 0.0234 | 0.7891 ± 0.0117 | 0.4333 ± 0.0342 |
| a_disagree | 0.1792 ± 0.0230 | 0.9018 ± 0.0154 | 0.6983 ± 0.0178 |

**Ini satu-satunya hasil positif Experiment 4** dan sampai sekarang belum pernah
diuji di run resmi. Aturan `mean` menurunkan EER di kedua varian
(−0.015 dan −0.021) dan menaikkan AUROC. Tetapi: hanya **40 query unknown**,
sehingga std ±0.02 dan satu query salah = 2.5 % EER. Bukti masih sangat lemah.

---

## 4. Status gerbang (sebagaimana dilaporkan exp4)

| Gerbang | Kriteria | whisper_l4 | whisper |
|---|---|---|---|
| G4.1 ceiling | Δ ≥ 0.02 → lanjut | +0.0225 → **LOLOS** | +0.0133 → **TUTUP** |
| G4.2 fusi | fusi > ECAPA-saja | 0.8408 < 0.8417 → **GAGAL** | 0.8325 < 0.8350 → **GAGAL** |
| G4.3 deteksi | aturan dua ruang < baseline | `mean` menang → **KANDIDAT** | `mean` menang → **KANDIDAT** |

Keputusan exp4: fusi Whisper ditutup atas dasar **G4.2**, bukan G4.1
(G4.1 sebenarnya lolos untuk `whisper_l4`).

---

## 5. Koreksi dari re-audit 15 Agustus 2026

Diukur ulang pada varian `whisper` (cache `whisper_l4` kosong):

| Ukuran | Nilai | Δ vs ECAPA |
|---|---|---|
| Oracle **seleksi** (dilaporkan exp4 sbg "ceiling") | 0.8508 | +0.0142 |
| Oracle **fusi linear** (w optimal per query) | 0.8700 | **+0.0333** |
| — query yang kedua argmin-nya salah tapi tertolong | 0.0192 | |
| w global terbaik (grid 0.01, bukan 6 titik) | 0.8392 @ w=0.883 | +0.0025 |

`delta_ceiling` adalah batas atas untuk **memilih salah satu ruang**, bukan untuk
**menjumlahkan skor**. Ceiling sebenarnya 2.3× lebih besar. Untuk `whisper`,
+0.0333 > 0.02 → G4.1 sebenarnya **LOLOS juga**, bukan tutup.

### Geometri ruang (1190 utterance / 111 speaker `base_train`)

| Metrik | ECAPA (192-d) | Whisper L3 (512-d) |
|---|---|---|
| ‖mean vektor unit‖ | 0.548 | **0.990** |
| rata-rata cosine antar-utterance | 0.299 | **0.981** |
| std cosine antar-utterance | 0.103 | **0.0062** |
| pangsa varians PC-1..10 | 27.9 % | 48.2 % |
| effective rank | 68.7 / 192 | **25.9 / 512** |
| Fisher ratio (antar/dalam speaker) | 1.10 | **0.51** |

### Operator fusi alternatif (ruang AS-Norm, varian `whisper`)

| Operator | Akurasi | Δ vs ECAPA |
|---|---|---|
| ECAPA saja | 0.8367 | — |
| weighted prob 0.9/0.1 | 0.8367 | +0.0000 |
| aturan min | 0.8225 | −0.0142 |
| sum rule (softmax) | 0.8108 | −0.0258 |
| product rule (softmax) | 0.7725 | −0.0642 |
| RRF (k=10) | 0.5392 | −0.2975 |
| RRF (k=60) | 0.5267 | −0.3100 |
| Borda rank sum | 0.5183 | −0.3183 |
| aturan max | 0.4717 | −0.3650 |

### Post-processing ruang Whisper (fit di `base_train`)

| Transform | Whisper saja | fusi terbaik | Δ vs ECAPA |
|---|---|---|---|
| raw (exp4) | 0.3125 | 0.8342 | −0.0025 |
| center | 0.3133 | 0.8342 | −0.0025 |
| **all-but-the-top k=1** | **0.3475** | 0.8342 | −0.0025 |
| all-but-the-top k=10 | 0.2783 | 0.8392 | +0.0025 |
| PCA-whiten d=128 | 0.3208 | 0.8358 | −0.0008 |
| **PCA-whiten penuh** | 0.3433 | **0.8417** | **+0.0050** |
| WCCN | 0.3417 | 0.8408 | +0.0042 |
| LDA d=128 | 0.2917 | 0.8400 | +0.0033 |

Transform yang sama diterapkan ke **ECAPA** (efek berdiri sendiri, tanpa fusi):

| Transform | ECAPA saja |
|---|---|
| raw | 0.8367 |
| **all-but-the-top k=1** | **0.8475** |
| all-but-the-top k=10 | 0.8442 |
| WCCN | 0.8317 |
| PCA-whiten d=128 | 0.8167 |
| LDA d=64 | 0.7375 |
| PCA-whiten d=64 | 0.7325 |

### Verifikasi konkatenasi `ScoreFusionEmbed`

| w | galat maks vs `2(w·cosdist_e + (1−w)·cosdist_w)` | deviasi unit-norm |
|---|---|---|
| 0.2 | 2.38e-06 | 5.96e-08 |
| 0.4 | 1.55e-06 | 0.0 |
| 0.7 | 2.62e-06 | 5.96e-08 |

Konkatenasi benar sampai presisi float32. Tidak ada bug penggabungan.

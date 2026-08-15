# Experiment 6 — Plan: ECAPA + Whisper, dengan readout Whisper yang benar

**Status:** rencana (belum dieksekusi)
**Tanggal:** 15 Agustus 2026
**Prasyarat baca:** [`experiment-4-evaluation-table.md`](experiment-4-evaluation-table.md),
[`experiment-4-reaudit.md`](experiment-4-reaudit.md)

---

## 0. Koreksi premis sebelum mulai

Dua hal perlu diluruskan supaya Experiment 6 tidak dibangun di atas asumsi keliru.

**(a) Experiment 4 bukan hasil evaluasi terbaik — ia bukan run evaluasi sama sekali.**
Exp4 adalah analisis gerbang: task paruh-validasi 100 speaker, prototype statis
1-shot, tanpa sesi inkremental, 3 seed, 40 query unknown, tanpa uji signifikansi.
Angka 0.8417 di sana adalah **ECAPA-saja**, bukan sistem. Run resmi terbaik saat
ini adalah **exp5b = 0.8760 ± 0.0160** (10 repetisi), disusul exp3b = 0.8332.
Detail lengkap ada di §0 dokumen tabel evaluasi.

**(b) Kegagalan fusi bukan cuma soal Whisper — exp5b juga gagal, dan itu tersembunyi.**
Angka final exp5b:

| Arm | Akurasi | vs A3 |
|---|---|---|
| A1 ECAPA saja | 0.8332 ± 0.0150 | p = 1.6e-06, A3 menang |
| **A2 ReDimNet saja** | **0.8766 ± 0.0174** | **p = 0.880 — tidak beda** |
| A3 fusi (diusulkan) | 0.8760 ± 0.0160 | — |

**A2 ≥ A3.** Seluruh kenaikan 0.8332 → 0.8760 berasal dari **mengganti backbone**,
bukan dari **menggabungkan**. Mekanisme fusi menyumbang nol — persis diagnosis yang
sama dengan Whisper di exp4, hanya tertutupi karena backbone barunya bagus.

Konsekuensi untuk Experiment 6: kriteria sukses **tidak boleh** "A3 > A1". Itu
sudah pasti tercapai kalau readout Whisper diperbaiki, dan tidak membuktikan apa
pun tentang fusi. Kriteria yang benar adalah **A3 > max(A1, A2) secara signifikan**.
Ini harus di-*pre-register* sekarang, sebelum melihat hasil.

---

## 1. Review celah Experiment 4

Diurutkan dari yang paling berdampak. Kolom "biaya" = estimasi kasar effort.

### Celah kritis (mengubah kesimpulan)

**C0 — Seluruh hasil Experiment 4 dihitung dari cache Whisper yang tidak valid.**
Ini celah terbesar dan sebelumnya tidak tercatat. Kronologi timestamp:
exp4 di-run **2026-07-18 18:52**; bug masked-pooling di `whisper_encoder.py`
diperbaiki **2026-07-26 18:21**; cache `whisper/` dihitung ulang
**2026-07-26 20:57–21:40**. Exp4 **tidak pernah** di-run ulang setelahnya, dan
cache `whisper_l4/` (varian terbaiknya) sampai sekarang kosong.

Bug-nya serius: mean-pool merata-ratakan seluruh 30 detik termasuk padding senyap,
sehingga klip 4 detik punya ~87 % embedding-nya didorong keheningan. Dampak
terukur pada protokol yang praktis identik:

| Varian `whisper` (L3), AS-Norm | exp4 (cache lama) | re-audit (cache baru) | Δ |
|---|---|---|---|
| ECAPA saja | 0.8350 | 0.8367 | +0.0017 *(noise protokol)* |
| **Whisper saja** | **0.2317** | **0.3125** | **+0.0808 (+35 % relatif)** |

ECAPA cocok dalam 0.0017 → protokolnya sepadan; selisih Whisper 35× lebih besar
dari noise itu. Jadi vonis "fusi Whisper ditutup" diambil dari embedding rusak.
Kesimpulan akhirnya mungkin bertahan (dengan cache baru pun fusi terbaik hanya
+0.005), tapi **argumennya harus dibangun ulang di atas angka yang benar**, dan
titik awal G6.1 bergeser dari 0.2425 ke 0.3125.
*Perbaikan:* pulihkan cache `whisper_l4`, run ulang exp4. *Biaya: 2–4 jam GPU
+ 10 menit run.*

**C1 — Definisi ceiling salah.**
`oracle_report.delta_ceiling` adalah oracle **seleksi** (pilih satu ruang per query),
tapi didokumentasikan sebagai *"the upper bound ANY fusion mechanism could reach"*.
Ceiling sebenarnya untuk `w·z_e+(1−w)·z_w` = **+0.0333** vs +0.0142 yang dilaporkan.
Akibatnya gerbang G4.1 (ambang 0.02) salah vonis untuk varian `whisper`.
*Perbaikan:* tambahkan `linear_fusion_oracle()` ke `src/evaluation/complementarity.py`
dan laporkan kedua angka. *Biaya: 1 jam* (kodenya sudah ada di
`scripts/exp4_reaudit_ceiling.py`).

**C2 — Readout Whisper adalah versi terlemah dari resep literatur.**
Satu layer beku, mean-pool saja, tanpa std, tanpa agregasi multi-layer, tanpa
backend terlatih. Akibatnya Fisher ratio 0.51 (variasi *dalam* speaker > *antar*
speaker) dan effective rank 26 dari 512. Whisper-PMFA dengan resep yang benar
mencapai EER 1.42 % di VoxCeleb1, **lebih baik 0.58 % absolut dari ECAPA-TDNN**.
*Perbaikan:* §2 fase F6-1. *Biaya: 1–2 hari* (termasuk recompute cache).

**C3 — Tidak ada centering/whitening di mana pun.**
Ruang Whisper anisotropik ekstrem (rata-rata cosine antar-utterance 0.981).
Sudah terbukti bisa diperbaiki: all-but-the-top k=1 menaikkan Whisper-sendiri
0.3125 → 0.3475. *Biaya: 2 jam.*

**C4 — Bobot fusi global dan tetap.**
Selisih antara +0.0025 (w global terbaik) dan +0.0333 (w optimal per query)
hanya bisa dipanen dengan bobot **adaptif**. Sweep manual tidak akan pernah
sampai ke sana. *Perbaikan:* LLR fusion (§2 fase F6-3). *Biaya: 1 hari.*

### Celah metodologis (tidak mengubah kesimpulan, tapi lemah di sidang)

**C5 — Grid sweep w terlalu kasar dan tidak simetris.**
Hanya 6 titik {0.5…0.95}, tidak ada w < 0.5. Dengan grid 0.01, optimum
sebenarnya ada di w = 0.883 → 0.8392, sedangkan grid exp4 melewatkannya.
*Biaya: 10 menit.*

**C6 — Hanya 40 query unknown untuk G4.3.**
Satu query salah = 2.5 % EER. Std terukur ±0.02–0.03. Temuan `mean` (EER
0.1654 → 0.1508) — satu-satunya hasil positif exp4 — berdiri di atas bukti
yang terlalu tipis untuk diklaim.

> **Koreksi rencana (15 Agustus, saat implementasi).** Draf awal dokumen ini
> menyarankan memakai **paruh-deteksi** (530 query). **Itu salah dan tidak
> jadi dilakukan.** Paruh-deteksi adalah populasi yang dipakai run resmi untuk
> menskor unknown-nya; memilih aturan deteksi di sana = memilih di atas test
> set (kebocoran). Desain exp4 yang asli — menjaga paruh-deteksi tetap
> pristine — memang benar.
>
> *Perbaikan yang dipakai:* panel kedua yang bebas bocor — unknown diambil dari
> speaker `base_train` yang **dikeluarkan dari cohort AS-Norm** (paruh speaker
> disjoint, disiplin yang sama dengan `split_cohort_and_genuine`). Hasilnya
> **592 query unknown / 56 speaker** dengan cohort 300 utt / 55 speaker yang
> tidak beririsan. Panel asli (40 query) tetap dilaporkan apa adanya supaya
> sebanding dengan exp4. *Biaya: 2 jam.*

**C7 — Tidak ada uji signifikansi sama sekali.**
Semua vonis gerbang diambil dari mean 3 seed tanpa CI atau p-value, padahal
selisih yang diperdebatkan (0.8408 vs 0.8417 = 0.0009) jauh lebih kecil dari
std antar-seed (0.0031). *Biaya: 2 jam* (`src/evaluation/statistics.py` sudah ada).

**C8 — 3 seed, prototype statis, tanpa dinamika continual.**
Hasil exp4 tidak transfer ke run resmi secara desain. Ini wajar untuk analisis
gerbang, tapi harus dinyatakan eksplisit di tesis, bukan dibiarkan implisit.
*Biaya: 0 (penulisan).*

### Celah operasional

**C9 — Cache `whisper_l4` kosong (0 file).**
Varian **terbaik** exp4 tidak reproducible saat ini. Re-audit terpaksa memakai
L3. Ini blocker untuk klaim apa pun tentang `whisper_l4`. *Biaya: 2–4 jam GPU.*

**C10 — Angka di branch `main` sudah usang.**
`experiments/full_evaluation_summary_*.json` di `main` masih 5-repetisi
pra-bugfix (exp5b 0.9078). Angka final 10-repetisi (0.8760) hanya ada di
`fix/recompute-exp3-exp5-bugfix` yang belum di-merge. Risiko mengutip angka
salah di tesis. *Perbaikan: merge branch.* *Biaya: 30 menit.*

**C11 — Parameter AS-Norm tidak pernah di-sweep di exp4.**
`top_k=200` dari cohort 300 berarti 67 % cohort dipakai — "adaptive" jadi
nyaris tidak adaptif. Diwarisi dari exp3b tanpa validasi ulang di konteks
dua-ruang. *Biaya: 3 jam.*

**C12 — Prototype 1-shot merugikan Whisper secara tidak proporsional.**
Ruang dengan Fisher ratio < 1 paling menderita saat prototype dibentuk dari
satu utterance. Belum pernah diuji apakah komplementaritas muncul di k-shot
lebih tinggi. *Biaya: 2 jam.*

**C13 — Temuan positif G4.3 menganggur.**
Aturan deteksi `mean` menang di kedua varian tapi tidak pernah masuk run resmi
maupun tag produksi. *Biaya: 1 hari* (masuk fase F6-4).

---

## 2. Rancangan Experiment 6

### Hipotesis

> Kegagalan fusi ECAPA+Whisper di Experiment 4 disebabkan oleh **readout Whisper**
> (satu layer, mean-pool, tanpa backend), bukan oleh **operator fusi**. Dengan
> readout gaya Whisper-PMFA — agregasi multi-layer + attentive statistics pooling
> + post-processing anti-anisotropi — ruang Whisper menjadi cukup diskriminatif
> sehingga fusi skor adaptif (LLR) mengungguli backbone terbaiknya sendiri.

### Kriteria sukses (pre-registered)

| | Kriteria | Alasan |
|---|---|---|
| **Utama** | A3 > max(A1, A2), paired t-test p < 0.05, 10 repetisi | Menghindari jebakan exp5b (A2 ≥ A3) |
| Sekunder | EER open-set A3 < min(EER A1, EER A2) | G4.3 belum pernah diuji layak |
| Minimum | Whisper-saja ≥ 0.45 di task validasi exp4 | Kalau readout tidak terbaiki, sisanya sia-sia |

### Fase

#### F6-0 — Kebersihan + **run ulang Experiment 4** (blocking, ~1 hari)
- Merge `fix/recompute-exp3-exp5-bugfix` ke `main` (**C10**).
- Recompute cache `whisper_l4` dengan pooling yang sudah diperbaiki (**C0**, **C9**).
- Tambahkan `linear_fusion_oracle()` ke `complementarity.py` + tes (**C1**).
- Perhalus grid sweep w ke 0.01 dan rentangkan ke [0, 1] (**C5**).
- Perbesar set unknown analisis ke paruh-deteksi (**C6**), tambahkan
  bootstrap CI ke semua gerbang (**C7**).
- **Run ulang `exp4_complementarity_asnorm.py` seutuhnya** pada cache yang benar,
  tulis ke `experiments/exp4_ceiling_asnorm_v2.json`. Ini menghasilkan tabel
  Experiment 4 yang sah untuk dikutip di tesis — yang sekarang ada tidak layak
  kutip (**C0**).

**Gerbang G6.0:** exp4 versi-2 (cache benar + metrik terkoreksi) direproduksi dan
konsisten dengan re-audit (+0.033 ceiling fusi-linear, Whisper-saja ~0.31 untuk L3).
Kalau `whisper_l4` versi-2 ternyata sudah menembus Whisper-saja ≥ 0.45 tanpa
readout baru, lompati F6-1 dan langsung ke F6-2.

#### F6-1 — Readout Whisper baru (~1–2 hari)
Backbone baru `whisper_pmfa` di `src/features/cache.py` (namespace cache sendiri,
semua tag lama tidak tersentuh):

1. Ambil hidden state dari **beberapa** blok encoder — mulai dari {3, 4, 5, 6}
   (Whisper-PMFA: blok tengah-ke-akhir menyimpan paling banyak info speaker).
2. Per layer, pooling **mean *dan* std** sepanjang sumbu waktu, dengan masking
   padding yang sudah ada (`_valid_encoder_frames`). → 512 × 2 × 4 = 4096 dim.
3. Konkat, lalu post-processing anti-anisotropi yang **di-fit di `base_train`**:
   all-but-the-top k=1 dan PCA-whiten (**C3**) — pilih via validasi.
4. Opsional reduksi ke ~256 dim via LDA `base_train` kalau 4096 terlalu berisik.

Semuanya **training-free** — tidak ada backend yang dilatih di tahap ini. Ini
sengaja: kalau readout training-free saja sudah menembus gerbang, kita hemat
berminggu-minggu; kalau tidak, baru pertimbangkan backend terlatih (F6-6).

**Gerbang G6.1:** Whisper-saja pada task validasi exp4 ≥ **0.45**
(baseline: 0.2425 untuk `whisper_l4`, 0.3125 untuk `whisper` + ABTT).
Ukur juga Fisher ratio — target > 1.0 (sekarang 0.51).

> **Kalau G6.1 gagal:** hipotesisnya salah, readout training-free tidak cukup.
> Pilihan: (a) lanjut ke F6-6 (backend terlatih, mahal), atau (b) hentikan jalur
> Whisper dan tulis exp6 sebagai *negative result* yang rapi — itu tetap
> kontribusi tesis yang sah, dan sekarang punya bukti geometris untuk
> menjelaskan **kenapa**.

#### F6-2 — Analisis komplementaritas ulang (~3 jam)
Jalankan `exp4_complementarity_asnorm.py` (versi terkoreksi F6-0) dengan
`--second-backbone whisper_pmfa`. Laporkan kedua oracle.

**Gerbang G6.2:** oracle fusi-linear ≥ **+0.03** di atas ECAPA-saja.
(Sekarang +0.0333 dengan Whisper yang *rusak*; readout yang benar harus
menaikkannya, bukan sekadar mempertahankannya.)

#### F6-3 — Fusi skor adaptif (LLR) (~1 hari)
Ganti `DualASNorm.weight` yang tetap dengan fusi regresi logistik gaya
BOSARIS/FoCal:

```
LLR_fused = a0 + a1·z_ecapa + a2·z_whisper  [+ a3·(fitur kualitas)]
```

Dilatih dengan cross-entropy berbobot prior pada trial `base_train`
(speaker-disjoint dari task — pakai `split_cohort_and_genuine` yang sudah ada
supaya tidak bocor). Ini jalur yang bisa memanen selisih +0.003 → +0.033 karena
bobot efektifnya bergantung pada trial. Implementasi sebagai
`fusion_strategy="llr"` di `src/experiments.py`, tag `exp6_llr_fusion`.

**Gerbang G6.3:** pada sweep validasi, A3 > max(A1, A2) dengan margin > 1 std
antar-seed.

#### F6-4 — Aturan deteksi dua ruang (~1 hari)
Uji ulang temuan G4.3 (`mean`) pada 530 query unknown paruh-deteksi (**C6**,
**C13**), bersama aturan LLR dari F6-3.

**Gerbang G6.4:** EER A3 < min(EER A1, EER A2), CI bootstrap tidak memotong nol.

#### F6-5 — Run resmi (~1 hari)
`scripts/run_full_evaluation.py` dengan tag `exp6_*`, 10 repetisi, ablasi penuh
A1/A2/A3/B1 + 3 baseline, paired t-test. Bandingkan langsung dengan exp5b.

#### F6-6 — (opsional, hanya kalau G6.1 gagal) Backend terlatih (~1–2 minggu)
Head gaya ECAPA/TDNN di atas fitur multi-layer Whisper, dilatih AAM-softmax di
`base_train`, dengan LoRA pada encoder Whisper. Ini resep Whisper-PMFA penuh.
**Risiko tinggi:** hanya 111 speaker `base_train` — Whisper-PMFA dilatih di
VoxCeleb2 (5994 speaker). Kemungkinan besar overfit. Jangan mulai tanpa
menambah data latih.

---

## 3. Ringkasan usaha & risiko

| Fase | Effort | Risiko | Bisa dibuang? |
|---|---|---|---|
| F6-0 kebersihan + exp4 v2 | ~1 hari | rendah | **tidak** — memperbaiki C0/C1/C5/C6/C7/C9/C10 |
| F6-1 readout | 1–2 hari | **sedang-tinggi** | tidak — inti hipotesis |
| F6-2 ceiling | 3 jam | rendah | tidak |
| F6-3 LLR fusion | 1 hari | sedang | tidak — inti kontribusi |
| F6-4 deteksi | 1 hari | rendah | ya, kalau waktu mepet |
| F6-5 run resmi | 1 hari | rendah | tidak |
| F6-6 backend | 1–2 minggu | **tinggi** | ya — hanya kalau G6.1 gagal |

**Total jalur utama: ~5 hari kerja.**

### Risiko terbesar

Skenario paling mungkin (~50 %) adalah **G6.1 lolos tapi G6.3 gagal**: readout
diperbaiki, Whisper-saja naik ke 0.5–0.6, tetapi fusi tetap tidak mengalahkan
backbone terbaiknya — sama seperti exp5b. Kalau itu terjadi, tesis punya **tiga**
bukti independen bahwa fusi dua-backbone tidak menambah apa pun di rezim
1-shot/few-shot open-set ini. Itu bukan kegagalan; itu temuan yang bisa
dipertahankan, asalkan kriteria suksesnya sudah di-*pre-register* sekarang dan
bukan digeser setelah melihat hasil.

Karena itu F6-0 (C1, C7) wajib dikerjakan lebih dulu: tanpa metrik ceiling yang
benar dan uji signifikansi, hasil negatif apa pun dari exp6 akan sama rapuhnya
dengan G4.1 di exp4.

---

## 4. Yang **tidak** akan dikerjakan

Sudah tertutup oleh bukti, jangan buang waktu:

- **Mengganti operator fusi.** Rank/Borda/RRF/product/min/max semua diuji,
  semuanya lebih buruk dari ECAPA-saja (tabel §5 dokumen evaluasi).
- **Bilinear/tensor fusion (TFN, LMF, MCB/MFB).** Menangani beda dimensi dengan
  elegan, tapi butuh ribuan speaker; dengan 111 speaker `base_train` pasti
  overfit. Cukup masuk *related work*.
- **Memperbaiki `GatedAttentionFusion`.** Interpolasi konveks di ruang 256-dim
  bersama; sudah terbukti merusak ruang ECAPA di exp1 dan tidak dipakai sejak
  exp2.
- **CCA/DCCA.** Versi linearnya (LDA/WCCN/PCA-whiten) sudah diuji, mentok
  +0.005. CCA tidak akan menciptakan informasi yang tidak ada di ceiling.
- **Mencari bug di konkatenasi.** Terverifikasi benar sampai presisi float32.

---

## 5. Referensi

- Zhao et al., [*Whisper-PMFA: Partial Multi-Scale Feature Aggregation for Speaker
  Verification using Whisper Models*](https://arxiv.org/abs/2408.15585) (2024) —
  dasar F6-1.
- Chen et al., [*WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
  Speech Processing*](https://arxiv.org/abs/2110.13900) (2021) — weighted-sum
  antar layer dengan bobot terlatih.
- [*MMFA: Masked Multi-Layer Feature Aggregation for Speaker Verification Using
  WavLM*](https://doi.org/10.3390/electronics14193857) (2025).
- Brümmer & de Villiers, *The BOSARIS Toolkit* (2013); Brümmer,
  [*Tutorial on logistic-regression calibration and fusion*](https://arxiv.org/pdf/2104.08846)
  (2021) — dasar F6-3.
- [*AMECxSV: Adaptive Metadata-Driven Embedding-Fusion Calibration*](https://arxiv.org/html/2607.16532)
  — quality-aware calibration.
- Mu & Viswanath, [*All-but-the-Top*](https://openreview.net/forum?id=HkuGJ3kCb)
  (ICLR 2018); Su et al., [*Whitening Sentence Representations*](https://arxiv.org/abs/2103.15316)
  (2021) — dasar C3.

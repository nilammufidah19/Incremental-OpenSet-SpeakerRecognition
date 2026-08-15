# Re-audit Experiment 4 — apakah penggabungan ECAPA + Whisper salah?

**Tanggal:** 15 Agustus 2026
**Skrip:** `scripts/exp4_reaudit_geometry.py`, `scripts/exp4_reaudit_ceiling.py`
**Artefak:** `experiments/exp4_reaudit_geometry.json`, `experiments/exp4_reaudit_ceiling.json`
**Protokol:** identik dengan Experiment 4 — 100 speaker paruh-validasi, 1-shot support,
4 query/speaker, 3 seed (0/1/2), AS-Norm cohort 300 / top_k 200 dari `base_train`.
Semua transformasi (centering/whitening/LDA/WCCN) **di-fit hanya pada `base_train`**
(speaker-disjoint dari task), jadi tidak ada kebocoran.

> **Catatan cakupan:** cache `data/cache/embeddings/whisper_l4/` kosong pada checkout ini
> (0 file), jadi re-audit dijalankan pada varian `whisper` (L3, `layer_fraction=0.5`).
> Di Experiment 4 kedua varian berperilaku sama (ceiling +0.0133 vs +0.0225;
> acc Whisper 0.2317 vs 0.2425), jadi kesimpulan kualitatif berlaku untuk keduanya,
> tetapi angka eksak untuk `whisper_l4` perlu di-recompute jika cache dipulihkan.

---

## 1. Ringkasan eksekutif

| Dugaan | Vonis |
|---|---|
| Bug pada **konkatenasi** dua embedding beda dimensi | **Tidak ada.** Identitas matematis `ScoreFusionEmbed` terverifikasi numerik sampai presisi float32 (galat maks 2.6e-6). |
| Bug pada **penurunan dimensi** | **Tidak relevan di exp4.** Exp4 tidak menurunkan dimensi sama sekali — tiap backbone tetap di ruang native-nya. Penurunan dimensi hanya ada di `GatedAttentionFusion` (exp1), dan itu memang sudah terbukti merusak ruang ECAPA. |
| Ada **kesalahan metodologis** | **Ya, satu, dan cukup serius** — definisi "ceiling" yang dipakai untuk menutup gerbang G4.1 bukan batas atas dari keluarga fusi yang sebenarnya diuji (§3). |
| Ada **kesalahan pengondisian ruang** | **Ya, tapi bukan penyebabnya.** Ruang Whisper sangat anisotropik dan tidak pernah di-center/whiten (§4); memperbaikinya menaikkan Whisper-sendiri 0.313→0.348 tapi tetap tidak membuka fusi. |
| **Kesimpulan praktis exp4** ("fusi Whisper ditutup") | **Tetap berdiri**, tapi alasannya harus diganti (§6). |

---

## 2. Konkatenasi itu benar — bukti numerik

`src/models/fusion.py::ScoreFusionEmbed` mengeluarkan
`out = [sqrt(w)·ê_ecapa ; sqrt(1−w)·ê_whisper]` (192 + 512 = 704 dim).
Klaim di docstring: jarak Euclid kuadrat antar dua vektor seperti itu persis sama dengan
`2·(w·cosdist_ecapa + (1−w)·cosdist_whisper)`.

Diuji pada tensor acak untuk w ∈ {0.2, 0.4, 0.7}:

| w | galat maks \|sq_euclid − target\| | deviasi maks dari unit-norm |
|---|---|---|
| 0.2 | 2.38e-06 | 5.96e-08 |
| 0.4 | 1.55e-06 | 0.0 |
| 0.7 | 2.62e-06 | 5.96e-08 |

Identitasnya eksak. Perbedaan dimensi 192 vs 512 **tidak** membuat Whisper
"mendominasi karena lebih panjang", karena tiap paruh sudah di-L2-normalisasi
lebih dulu sehingga tiap paruh menyumbang tepat `w` dan `1−w` ke skor akhir.

Hal yang sama berlaku untuk `DualASNorm` (exp5): ia memecah di `split_dim`,
menormalkan tiap paruh di ruangnya sendiri, lalu menjumlah z-score berbobot.
Faktor skala konstan `sqrt(w)` hilang di z-score — sudah ada tesnya di
`tests/test_experiment5.py`.

**Satu-satunya tempat dimensi benar-benar diturunkan** adalah
`GatedAttentionFusion`: 192→256 dan 512→256, lalu digabung *konveks*
`g⊙e'_1 + (1−g)⊙e'_2`. Ini bukan konkatenasi melainkan **interpolasi** — kedua
cabang dipaksa hidup di sistem koordinat 256-dim yang sama dan saling berebut
unit yang sama, sehingga kapasitasnya tidak bertambah. Inilah yang bikin A1
kolaps ke ~16% di diagnosa exp1, dan alasan `residual_init` harus ditambahkan
(bias gate ke sigmoid(4)≈0.982 supaya Whisper cuma jadi residual kecil).
Desain ini memang lemah, tapi exp4 sudah benar tidak memakainya.

---

## 3. Kesalahan metodologis: "ceiling" yang dipakai bukan batas atas fusi

`src/evaluation/complementarity.py::oracle_report` menghitung

```
acc_oracle = mean( argmin(d_ecapa) benar  ATAU  argmin(d_whisper) benar )
delta_ceiling = acc_oracle − acc_ecapa
```

dan docstring `scripts/exp4_complementarity_asnorm.py` menyebutnya
*"the upper bound ANY fusion mechanism could reach"*.

**Itu tidak benar.** Angka itu adalah oracle dari **pemilihan (selection)** —
batas atas kalau kita boleh memilih *salah satu* ruang per query. Padahal fusi
yang benar-benar diuji di 4b adalah **penjumlahan berbobot**
`w·z_e + (1−w)·z_w`, dan penjumlahan bisa benar pada query di mana **kedua**
argmin salah (kelas benar peringkat-2 di kedua ruang, tapi menang setelah
dijumlah). Jadi oracle seleksi bukan batas atas untuk keluarga itu.

Batas atas yang benar dihitung eksak di `scripts/exp4_reaudit_ceiling.py`:
untuk tiap query, cari apakah **ada** w ∈ [0,1] yang membuat kelas benar jadi
argmin. Kondisinya linear dalam w, jadi tiap kompetitor memberi setengah-garis
dan tinggal diiris:

| Ukuran (ruang AS-Norm, rata-rata 3 seed) | Nilai | Δ vs ECAPA |
|---|---|---|
| ECAPA saja | 0.8367 | — |
| Whisper saja | 0.3125 | — |
| **Oracle SELEKSI** (yang dipakai exp4 sbg ceiling) | 0.8508 | **+0.0142** |
| **Oracle FUSI LINEAR** (w optimal per query) | 0.8700 | **+0.0333** |
| — di antaranya query yang kedua argmin-nya salah | 0.0192 | |
| w global terbaik tunggal (grid 0.01) | 0.8392 @ w=0.883 | +0.0025 |

Konsekuensi untuk tesis: ambang gerbang G4.1 adalah `delta_ceiling < 0.02 ⇒ tutup`.
Diukur dengan ceiling yang benar, angkanya **+0.0333 > 0.02**, artinya
**G4.1 seharusnya LOLOS, bukan tertutup.** Gerbang yang benar-benar menutup fusi
adalah G4.2 (sweep w) — dan itu tetap gagal: w global terbaik hanya +0.0025,
di dalam noise antar-seed.

Jadi kesimpulan akhirnya sama, tapi **argumennya harus diperbaiki** sebelum
sidang: klaim "ceiling adalah batas atas fusi apa pun" mudah dipatahkan penguji.
Kalimat yang aman: *"oracle seleksi +0.014 dan oracle fusi-linear +0.033
menunjukkan komplementaritas ada tetapi hanya bisa dipanen dengan bobot
per-query (adaptif); dengan bobot global tunggal — satu-satunya yang bisa
dikalibrasi tanpa label uji — perolehannya +0.003 dan tidak signifikan."*

---

## 4. Kesalahan pengondisian: ruang Whisper hampir kolaps

Geometri kedua ruang pada 1190 utterance / 111 speaker `base_train`:

| Metrik | ECAPA (192-d) | Whisper L3 (512-d) |
|---|---|---|
| ‖mean vektor unit‖ (0 = isotropik, 1 = kolaps) | 0.548 | **0.990** |
| rata-rata cosine antar-utterance | 0.299 | **0.981** |
| simpangan baku cosine antar-utterance | 0.103 | **0.0062** |
| pangsa varians PC-1 | 6.1 % | 13.1 % |
| pangsa varians PC-1..10 | 27.9 % | 48.2 % |
| effective rank (participation ratio) | 68.7 / 192 | **25.9 / 512** |
| Fisher trace ratio (antar-speaker / dalam-speaker) | 1.10 | **0.51** |

Ini "cone effect" klasik pada fitur transformer yang di-mean-pool: semua
embedding menunjuk ke arah yang nyaris sama, seluruh sinyal diskriminatif
hidup di pita cosine selebar ±0.006, dan dari 512 dimensi nominal hanya ~26
yang benar-benar terpakai. Fisher ratio < 1 artinya variasi **dalam** satu
speaker lebih besar daripada variasi **antar** speaker — Whisper L3 memang
bukan ruang speaker.

Pipeline sekarang tidak pernah melakukan centering maupun whitening
(`whisper_encoder.extract_embedding` cuma mean-pool lalu L2-normalisasi).
Itu memang kelalaian dibanding praktik standar. Tapi memperbaikinya tidak
menyelamatkan fusi:

| Transform (fit di base_train) | Whisper sendiri | ceiling seleksi | fusi terbaik | Δ vs ECAPA |
|---|---|---|---|---|
| raw (exp4 sekarang) | 0.3125 | +0.0142 | 0.8342 | −0.0025 |
| center | 0.3133 | +0.0108 | 0.8342 | −0.0025 |
| **all-but-the-top k=1** | **0.3475** | +0.0142 | 0.8342 | −0.0025 |
| all-but-the-top k=10 | 0.2783 | +0.0108 | 0.8392 | +0.0025 |
| PCA-whiten d=128 | 0.3208 | +0.0108 | 0.8358 | −0.0008 |
| **PCA-whiten penuh** | 0.3433 | +0.0125 | **0.8417** | **+0.0050** |
| WCCN | 0.3417 | +0.0133 | 0.8408 | +0.0042 |
| LDA d=128 | 0.2917 | +0.0142 | 0.8400 | +0.0033 |

Perbaikan terbaik untuk Whisper-sendiri: **0.3125 → 0.3475** (+11 % relatif)
hanya dengan membuang 1 komponen utama. Tapi perolehan fusi maksimum tetap
**+0.005**, masih di dalam noise. Artinya: pengondisian yang buruk itu nyata,
tapi **bukan** penyebab fusi gagal.

Catatan: transformasi yang sama diterapkan ke ECAPA justru **merusaknya**
(PCA-whiten d=64 → 0.7325, LDA d=64 → 0.7375, dari 0.8367). ECAPA sudah
dilatih diskriminatif, jadi whitening malah memperkuat arah bervarians rendah
yang isinya noise. Satu-satunya yang membantu ECAPA sedikit: all-but-the-top
k=1 (0.8367 → 0.8475). Ini kandidat murah yang berdiri sendiri, terlepas dari
soal fusi.

---

## 5. Operator fusi non-linear juga tidak menolong

Semua dihitung di ruang AS-Norm yang sama:

| Operator | Akurasi | Δ vs ECAPA |
|---|---|---|
| Borda / penjumlahan rank | 0.5183 | −0.3183 |
| Reciprocal Rank Fusion (k=10) | 0.5392 | −0.2975 |
| RRF (k=60) | 0.5267 | −0.3100 |
| aturan min | 0.8225 | −0.0142 |
| aturan max | 0.4717 | −0.3650 |
| product rule (softmax) | 0.7725 | −0.0642 |
| sum rule (softmax) | 0.8108 | −0.0258 |
| weighted prob 0.9/0.1 | 0.8367 | +0.0000 |

Fusi berbasis rank hancur karena rank membuang informasi magnitudo — persis
informasi yang bikin ECAPA menang. Tidak ada operator yang lolos.

---

## 6. Diagnosis akhir

Masalahnya **bukan di cara menggabungkan**, melainkan di **cara membaca
Whisper**. Pipeline sekarang mengambil *satu* layer encoder beku, mean-pool
sepanjang waktu, L2-normalisasi, selesai. Tiga hal yang hilang dibanding
praktik literatur:

1. **Satu layer saja.** Informasi speaker di Whisper tersebar di beberapa blok
   encoder; memilih satu layer membuang sisanya.
2. **Mean-pooling saja.** Statistik orde-2 (standar deviasi sepanjang waktu)
   membawa banyak informasi speaker dan seluruhnya dibuang.
3. **Tidak ada backend terlatih.** Tidak ada yang pernah mengoptimalkan ruang
   Whisper untuk diskriminasi speaker — Fisher ratio 0.51 adalah akibat
   langsungnya.

Ini juga menjelaskan kenapa Experiment 5 (ReDimNet-b2, backbone yang memang
dilatih untuk speaker) langsung berhasil di mana Whisper gagal.

---

## 7. Riset: metode penggabungan dua embedding beda dimensi

Dikelompokkan dari yang paling relevan untuk tesis ini.

### A. Score-level / calibration fusion — *paling relevan, standar di bidang ini*

Fusi di level skor sama sekali **tidak peduli dimensi** — itulah kenapa ini
jadi standar di NIST SRE. Bedanya dengan yang dipakai sekarang: bobotnya
**dipelajari lewat regresi logistik**, bukan di-sweep manual, dan bisa
memasukkan variabel kualitas sehingga bobotnya efektif adaptif per-trial —
persis yang dibutuhkan untuk memanen +0.033 di §3.

- Brümmer & de Villiers, *The BOSARIS Toolkit: Theory, Algorithms and Code for
  Surviving the New DCF* (2013) — fusi & kalibrasi LLR via linear logistic
  regression, plus optimizer cepat. Pendahulunya: FoCal.
- Brümmer, *[Tutorial on logistic-regression calibration and fusion](https://arxiv.org/pdf/2104.08846)*
  (2021) — turunan lengkap `LLR_fused = a0 + Σ a_i · s_i`, dilatih dengan
  cross-entropy berbobot prior.
- Ferrer et al., *[A comparison of linear and non-linear calibrations for speaker
  recognition](https://arxiv.org/pdf/1402.2447)* — kapan kalibrasi non-linear
  layak.
- *[AMECxSV: Adaptive Metadata-Driven Embedding-Fusion Calibration](https://arxiv.org/html/2607.16532)*
  — quality-aware calibration, bobot fusi jadi fungsi kondisi sinyal.

**Untuk tesis:** ini jalur paling murah dan paling defensible. `DualASNorm`
sudah menghasilkan dua z-score per trial; tinggal ganti `weight` tetap dengan
LLR terlatih di `base_train`.

### B. Perataan subruang (subspace alignment) — CCA / LDA / WCCN

Memproyeksikan dua ruang beda dimensi ke subruang bersama yang
memaksimalkan korelasi. Menangani dimensi berbeda secara alami (proyeksi
`W_1 ∈ R^{192×k}`, `W_2 ∈ R^{512×k}`).

- Andrew et al., *Deep Canonical Correlation Analysis* (ICML 2013) — versi
  non-linear dari CCA.
- Gao et al., *[Discriminative Multiple Canonical Correlation Analysis for
  Information Fusion](https://arxiv.org/pdf/2103.00361)* — DMCCA, versi
  class-aware; menunjukkan fusi berbasis CCA konsisten mengalahkan konkatenasi
  polos.
- *[Multi-Feature Learning with Canonical Correlation Analysis Constraint for
  Text-Independent Speaker Verification](https://ieeexplore.ieee.org/document/9383541/)*
  (ICASSP 2021) — CCA sebagai *constraint* saat melatih speaker embedding.

**Catatan dari data kita:** §4 sudah menguji versi linearnya (LDA, WCCN,
PCA-whiten) dan tidak ada yang menembus +0.005. CCA tidak akan menciptakan
informasi yang tidak ada di oracle +0.033.

### C. Bilinear / tensor fusion — interaksi multiplikatif

Outer product `e_1 ⊗ e_2` menangani dimensi berbeda secara langsung, tapi
menghasilkan 192×512 = 98k parameter per unit output; karena itu semua varian
modern memakai dekomposisi low-rank.

- Zadeh et al., *Tensor Fusion Network* (EMNLP 2017).
- Liu et al., *[Efficient Low-rank Multimodal Fusion (LMF)](https://arxiv.org/abs/1806.00064)*
  (ACL 2018) — dekomposisi low-rank, menghindari produk Cartesian eksplisit.
- Fukui et al., *Multimodal Compact Bilinear Pooling* (MCB, EMNLP 2016);
  Kim et al., *MLB* (ICLR 2017); Yu et al., *MFB* (ICCV 2017) — silsilah
  low-rank bilinear di VQA.

**Untuk tesis:** butuh data latih besar; dengan 111 speaker `base_train` ini
hampir pasti overfit. Sebut di *related work*, jangan diimplementasikan.

### D. Attention / gated fusion

- Arevalo et al., *Gated Multimodal Units* (ICLR-W 2017) — inilah keluarga
  `GatedAttentionFusion` yang sudah ada di `src/models/fusion.py`.
- *[Feature Alignment Determines Fusion Strategy: A Comparative Study of
  Cross-Attention and Concatenation in Multimodal Learning](https://arxiv.org/html/2606.01207v1)*
  — temuan kunci: cross-attention menang **hanya** kalau fiturnya tidak
  ter-align; kalau sudah align, konkatenasi sama baiknya dan jauh lebih murah.
  Ini mendukung keputusan exp4 memakai konkatenasi/score-fusion.

### E. Jalur yang sebenarnya dipakai literatur Whisper/SSL — *rekomendasi terkuat*

Alih-alih menggabung **dua backbone**, literatur menggabung **banyak layer di
dalam satu backbone**. Ini menjawab langsung ketiga kelemahan di §6.

- Zhao et al., *[Whisper-PMFA: Partial Multi-Scale Feature Aggregation for
  Speaker Verification using Whisper Models](https://arxiv.org/abs/2408.15585)*
  (2024) — agregasi *subset* blok encoder Whisper (blok tengah-ke-akhir
  menyimpan paling banyak info speaker), bukan satu layer. **EER 1.42 % di
  VoxCeleb1, 0.58 % absolut lebih baik dari baseline ECAPA-TDNN**, dan 8.23 %
  di CN-Celeb1. Dengan LoRA, parameter terlatih turun ~45× dengan penalti EER
  hanya 0.2 %. Ini bukti langsung bahwa Whisper **bisa** mengalahkan ECAPA —
  kalau readout-nya benar.
- Chen et al., *[WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
  Speech Processing](https://arxiv.org/abs/2110.13900)* (2021) — weighted-sum
  antar layer dengan bobot terlatih (gaya SUPERB) + downstream ECAPA-TDNN;
  >35 % perbaikan EER relatif atas ECAPA-TDNN berbasis Fbank.
- *[MMFA: Masked Multi-Layer Feature Aggregation for Speaker Verification Using
  WavLM](https://doi.org/10.3390/electronics14193857)* (2025) — generasi
  terbaru dari ide yang sama.

Pola bersamanya: **konkat/weighted-sum antar layer → attentive statistics
pooling (mean *dan* std) → backend ECAPA/TDNN terlatih dengan AAM-softmax.**
Yang dipakai proyek ini sekarang — satu layer, mean-pool, tanpa backend —
adalah versi paling lemah dari resep itu.

### F. Pra-pemrosesan anti-anisotropi

- Mu & Viswanath, *[All-but-the-Top: Simple and Effective Postprocessing for
  Word Representations](https://openreview.net/forum?id=HkuGJ3kCb)* (ICLR 2018)
  — kurangi mean, buang top-k komponen utama. Persis yang diuji di §4:
  naik +0.035 untuk Whisper, +0.011 untuk ECAPA.
- Su et al., *[Whitening Sentence Representations for Better Semantics and
  Faster Retrieval](https://arxiv.org/abs/2103.15316)* (2021) — whitening penuh
  untuk embedding transformer.
- *[Examining the effect of whitening on static and contextualized word
  embeddings](https://openreview.net/pdf?id=nrvtnXa_BJs)* — kapan whitening
  membantu dan kapan merusak; sejalan dengan temuan kita bahwa whitening
  merusak ECAPA tapi membantu Whisper.

---

## 8. Rekomendasi

**Prioritas 1 — perbaiki narasi, bukan kode (murah, wajib sebelum sidang).**
Ubah docstring `scripts/exp4_complementarity_asnorm.py` dan
`src/evaluation/complementarity.py::oracle_report`: `delta_ceiling` adalah
ceiling **seleksi**, bukan "batas atas fusi apa pun". Laporkan kedua angka
(+0.014 seleksi, +0.033 fusi-linear) di `docs/experiment-4.md`, dan pindahkan
beban penutupan dari G4.1 ke G4.2.

**Prioritas 2 — all-but-the-top k=1 sebagai post-processing berdiri sendiri.**
Ini menaikkan ECAPA-sendiri 0.8367 → 0.8475 di sweep validasi, tanpa melatih
apa pun dan tanpa menyentuh fusi. Perlu diverifikasi di run resmi 10-repetisi
sebelum diklaim.

**Prioritas 3 — kalau fusi mau dibuka lagi: ganti bobot tetap dengan LLR fusion**
(§7A) di atas `DualASNorm` yang sudah ada. Ini satu-satunya jalur yang bisa
memanen selisih antara +0.003 (bobot global) dan +0.033 (bobot per-query),
dan bisa dilatih di `base_train` tanpa kebocoran.

**Jangan** habiskan waktu mengganti operator fusi (§5 sudah menutup semuanya)
atau membangun bilinear/tensor fusion (§7C, datanya terlalu kecil).

**Kalau Whisper mau dihidupkan lagi**, ganti *readout*-nya mengikuti
Whisper-PMFA (§7E) — multi-layer + attentive stat pooling + backend terlatih —
bukan operator fusinya. Ini bukan perbaikan kecil: butuh recompute seluruh
cache embedding dan melatih backend, jadi realistis hanya kalau ada waktu
sisa setelah exp5 selesai.

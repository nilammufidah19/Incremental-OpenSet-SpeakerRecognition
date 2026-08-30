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

### Batasan bebas-pelatihan — dan di mana Experiment 6 berdiri terhadapnya

Tesis ini punya batasan tetap: **strict 1-shot, bebas-pelatihan (0 episode,
backbone beku, tanpa fine-tuning)** ([`experiment-5.md`](experiment-5.md) §16).
Terverifikasi di artefak: `n_train_episodes = 0` di semua run resmi.
`docs/experiment-5.md` §78 bahkan memakai batasan ini untuk menolak jalur
Whisper-untuk-SV yang berhasil di literatur (WSI 2025, Whisper-SV 2024),
karena semuanya melatih ulang/fine-tune backend.

Ini **membatasi Experiment 6 secara langsung** dan harus eksplisit:

| Komponen exp6 | Melatih sesuatu? | Status |
|---|---|---|
| F6-1 readout multi-layer + mean/std pooling | tidak — murni aritmatika di atas encoder beku | ✅ patuh |
| F6-1 post-processing (ABTT / whitening / LDA) | tidak ada gradien; hanya statistik yang di-*fit* di `base_train` | ✅ patuh — sekelas dengan cohort AS-Norm |
| F6-3 LLR fusion (regresi logistik) | ya — **3–4 koefisien skalar**, di-fit di trial `base_train` | ⚠️ **butuh keputusan** (lihat di bawah) |
| F6-3b fusi adaptif nol-parameter | tidak — tidak ada yang di-*fit* sama sekali | ✅ patuh di bacaan mana pun |
| F6-6 backend terlatih (AAM-softmax + LoRA) | ya — jutaan parameter, gradien pada encoder | ❌ **melanggar batasan** |

**Soal F6-3 — ajukan sebagai keputusan definisi, bukan permintaan izin.**
"Boleh pakai LLR?" adalah pertanyaan yang mudah dijawab "tidak" tanpa
memeriksa konsekuensinya. Yang harus diajukan ke pembimbing adalah **di mana
garisnya**, dengan tiga tingkat berikut:

| Tingkat | Definisi operasional | Isi sistem sekarang |
|---|---|---|
| **0 — representasi** | ada gradien pada parameter yang **menghasilkan embedding** | kosong. F6-6 (LoRA + AAM-softmax) ada di sini → dicoret |
| **1 — statistik & kalibrasi skor** | di-*fit* di `base_train` speaker-disjoint, **tidak menyentuh embedding**, O(10) parameter | cohort AS-Norm (300 utt); ambang `target_frr` (−1.5376 di exp5b); bobot `w=0.5` yang dipilih di validation sweep. **F6-3 LLR masuk ke sini** |
| **2 — nol parameter** | adaptif per-trial tanpa ada yang di-*fit* sama sekali | kosong — diisi F6-3b (lihat §Fase) |

Pertanyaan yang diajukan: **garisnya antara 0–1, atau antara 1–2?**

- Kalau **0–1** → F6-3 boleh, dan sistem sekarang tetap sah.
- Kalau **1–2** → F6-3 dilarang, **tapi ikut membatalkan AS-Norm dan ambang
  kalibrasi**, artinya exp3b dan exp5b (hasil headline tesis) ikut gugur.

Konsekuensi kedua itu harus dilihat pembimbing **sebelum** memutuskan, bukan
sesudah. Inilah bentuk argumennya yang paling kuat: pertanyaannya bukan
"apakah LLR pelatihan", melainkan "sistem yang sudah berjalan ini ada di sisi
mana". Literatur speaker verification menyebut tingkat 1 sebagai *calibration
and fusion*, bukan *training* (Brümmer dkk., IEEE TASLP 2007).

**Kalau F6-3 ditolak, jangan mundur ke `w` tetap** (§3: mentok +0.0025).
Yang dijalankan adalah **F6-3b**, fusi adaptif tingkat-2 tanpa satu pun
parameter yang di-fit — aman di bacaan batasan yang paling ketat sekalipun,
dan tetap adaptif per trial sehingga bisa memanen sebagian dari plafon
+0.0333. F6-3b **dikerjakan lebih dulu**, tanpa menunggu keputusan siapa pun.

**Urutan waktu (penting): keputusan ini tidak boleh memblokir apa pun.**
Memo tiga-tingkat dikirim sekarang → F6-1 dan F6-2 jalan paralel (keduanya
tidak bergantung padanya) → F6-3b dikerjakan sambil menunggu. Tidak ada fase
yang menganggur menunggu jawaban.

**F6-6 dicoret dari jalur utama.** Sebelumnya ditandai "opsional, risiko
tinggi"; dengan batasan ini ia bukan sekadar berisiko — ia **melanggar premis
tesis**. Jangan dikerjakan kecuali pembimbing secara eksplisit melonggarkan
batasan bebas-pelatihan. Konsekuensinya: kalau G6.1 gagal, tidak ada rencana
cadangan teknis — yang tersisa adalah menulis exp6 sebagai temuan negatif.

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

### Disiplin feature-flag (mengikat untuk semua fase)

Sama seperti exp1–exp5: **tidak ada perubahan perilaku yang masuk tanpa flag.**
Mekanismenya sudah ada dan tidak perlu dibuat baru — registry `ExperimentConfig`
+ `ACTIVE_EXPERIMENT` di [`src/experiments.py`](../src/experiments.py), yang
dibaca `scripts/run_full_evaluation.py` lewat `active()`.

Aturan yang berlaku untuk F6-1..F6-5:

1. **Knob baru = field baru di `ExperimentConfig`, dengan default = perilaku
   lama.** Setelah field ditambahkan, exp1/exp3b/exp5b harus tetap menghasilkan
   angka yang identik tanpa diedit. Ini presedennya `score_norm="none"`,
   `per_config_calibration=False`, `dump_detection_scores=False`.
2. **Setiap konfigurasi yang dilaporkan = satu entri `EXPERIMENTS`** dengan id
   stabil `exp6_*`, plus komentar blok di atasnya yang mencatat gerbang mana
   yang lolos/gagal dan file artefak JSON-nya (format yang sama dengan blok
   `exp5b_redimnet_fusion`).
3. **Backbone baru = namespace cache sendiri** di `BACKBONE_MODES` /
   `BACKBONE_EXTRACTORS` ([`src/features/cache.py`](../src/features/cache.py)).
   Cache `whisper`, `whisper_l4`, `redimnet_b2` tidak boleh disentuh —
   preseden `whisper_l4` sudah menunjukkan polanya.
4. **Script analisis: aditif saja.** Semua key output yang sudah ada
   mempertahankan artinya, knob baru lewat argumen CLI ber-default. Ini
   disiplin yang sudah dipakai di F6-0 supaya angka screening G5.1/G5.2 exp5
   tetap reproducible dari script yang sama.
5. **Rollback = flip satu tag**, bukan `git revert`. Kalau sebuah fase gagal
   gerbangnya, entri `exp6_*`-nya tetap tinggal di registry sebagai jejak
   temuan negatif — jangan dihapus.

| Fase | Flag / field | Default (= perilaku lama) | Tag run |
|---|---|---|---|
| F6-1 readout | `whisper_backbone="whisper_pmfa"`; field baru `whisper_layers`, `whisper_pooling`, `embedding_postproc` | `"whisper"`; `(3,)`, `"mean"`, `"none"` | `exp6_pmfa_readout` |
| F6-2 ceiling | tidak ada field runtime — CLI `--second-backbone whisper_pmfa` | `whisper` | — (script analisis) |
| F6-3 fusi LLR | `fusion_strategy="llr"` (nilai baru pada field yang sudah ada) + `llr_fit_split` | `"score_norm"`; `"base_train"` | `exp6_llr_fusion` |
| F6-3b adaptif nol-parameter | `fusion_strategy="adaptive_margin"` / `"rank"` | `"score_norm"` | `exp6_adaptive_fusion` |
| F6-4 deteksi | hanya kalau aturan diadopsi ke jalur runtime: `detection_rule` | `"min"` (perilaku sekarang); default-nya tetap post-hoc dari `dump_detection_scores` | `exp6_detect_*` |
| F6-5 run resmi | entri gabungan berisi flag F6-1+F6-3 yang lolos gerbang | — | `exp6_full` |

Catatan F6-1: `embedding_postproc` (ABTT / PCA-whiten / LDA) di-*fit* di
`base_train` dan disimpan bersama cache, bukan di-hardcode — supaya bisa
dimatikan (`"none"`) untuk ablasi tanpa recompute embedding.

### Soal beda dimensi (192 vs 512) — sudah tertutup, jangan dibuka lagi

Pertanyaan "ECAPA dinaikkan ke berapa, Whisper diturunkan ke berapa" berulang
kali muncul. Jawabannya sudah lengkap di
[`experiment-4-reaudit.md`](experiment-4-reaudit.md) §2 dan §4; diringkas di
sini supaya F6-1 tidak mengulang jalur yang sudah mati.

**(a) Proyeksi terlatih 192→256 / 512→256 — sudah dicoba, gagal, dan sekarang
juga melanggar batasan.** `GatedAttentionFusion` memakai dua `nn.Linear` plus
gate sigmoid (keluarga *Gated Multimodal Unit*, Arevalo dkk. ICLR-W 2017 /
NCAA 2020). Hasilnya: arm ECAPA-saja runtuh **0.79 → 0.16**
(`scripts/diagnose_accuracy_gap.py`) — akar akurasi 0.235 di exp0. Penyebab
strukturalnya: itu **interpolasi konveks, bukan konkatenasi**, jadi kedua
cabang berebut 256 unit yang sama dan kapasitasnya tidak bertambah. Ditambal
`residual_init`, lalu ditinggalkan. 312.064 parameter terlatih itu kini juga
melanggar batasan bebas-pelatihan — mati karena **dua** alasan independen.

**(b) Sejak exp2 tidak ada penaikan/penurunan dimensi sama sekali.** Fusi
terjadi di ruang natif masing-masing lewat konkatenasi berbobot
`[√w·ê₁ ; √(1−w)·ê₂]`, yang jarak Euclidean kuadratnya **persis**
`2(w·cosdist₁ + (1−w)·cosdist₂)` — galat maks terukur **2.6e-06**
(§2 re-audit). Karena tiap paruh di-L2-normalisasi lebih dulu
(*length normalization*: Garcia-Romero & Espy-Wilson, Interspeech 2011),
**beda dimensi tidak membuat Whisper mendominasi**. Perbedaan dimensi memang
sengaja dihindari, bukan terlewat.

**(c) Yang benar-benar diukur adalah pengondisian ruang**, bukan penyamaan
dimensi — `scripts/exp4_reaudit_geometry.py` →
`experiments/exp4_reaudit_geometry.json`, 8 transform di-fit di 4000 utterance
`base_train`:

| Transform | Whisper-sendiri | Δ fusi vs ECAPA | Efek ke ECAPA |
|---|---|---|---|
| raw | 0.3125 | −0.0025 | 0.8367 |
| **all-but-the-top k=1** | **0.3475** | −0.0025 | **0.8475** ✅ |
| PCA-whiten penuh | 0.3433 | +0.0050 | rusak (d=64 → 0.7325) |
| WCCN | 0.3417 | +0.0042 | — |
| LDA d=128 | 0.2917 ↓ | +0.0033 | rusak (d=64 → 0.7375) |

Kesimpulan yang mengikat F6-1: **ABTT k=1 default**, dan **fusi maksimum dari
seluruh keluarga transform ini tetap +0.005** — pengondisian buruk itu nyata,
tapi bukan penyebab fusi gagal.

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

> **Koreksi 2026-08-30 dari data kita sendiri** (`experiments/exp4_reaudit_geometry.json`,
> §4 re-audit): urutan langkah 3–4 di atas menyalahi bukti yang sudah ada.
> Pada ruang Whisper L3, **all-but-the-top k=1** adalah satu-satunya transform
> yang menaikkan Whisper-sendiri (0.3125 → 0.3475) *dan* satu-satunya yang
> tidak merusak ECAPA (0.8367 → 0.8475). **LDA justru menurunkan** Whisper
> (d=128 → 0.2917), dan PCA-whiten/LDA merusak ECAPA parah (d=64 → 0.7325 /
> 0.7375). Jadi: **default F6-1 = ABTT k=1**; PCA-whiten penuh sebagai ablasi
> kedua; **LDA di-pre-register sebagai ablasi yang diperkirakan gagal**, bukan
> sebagai jalur utama. Kalau LDA menang di ruang PMFA 4096-d, itu temuan yang
> berlawanan dengan bukti sekarang dan harus dilaporkan sebagai itu.

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

Ini jalur yang bisa memanen selisih +0.003 → +0.033 karena bobot efektifnya
bergantung pada trial. Implementasi sebagai `fusion_strategy="llr"` di
`src/experiments.py`, tag `exp6_llr_fusion`.

**Spesifikasi yang mengikat** — tiap butir ada supaya hasilnya bertahan di
sidang, bukan sekadar supaya jalan:

1. **Mulai dari 3 koefisien saja** (`a₀, a₁, a₂`). Fitur kualitas (`a₃`)
   hanya ditambahkan kalau versi 3-koefisien sudah lolos G6.3 — menambah
   parameter untuk mengejar gerbang yang gagal adalah cara tercepat kehilangan
   argumen "ini kalibrasi, bukan pelatihan".
2. **Split fit**: trial `base_train` yang speaker-disjoint dari task **dan
   dari cohort AS-Norm**. Kalau cohort dan fit-split beririsan, koefisiennya
   belajar dari statistik yang menormalkan skor input-nya sendiri
   (*double-dipping*). Pakai `split_cohort_and_genuine`.
3. **Fit ulang per repetisi**, memakai split `base_train` repetisi itu —
   bukan sekali fit lalu dipakai untuk 10 repetisi.
4. **Objektif**: prior-weighted cross-entropy + regularisasi L2.
5. **Laporkan Cllr dan minCllr**, bukan hanya akurasi. Itu bahasa standar
   literatur kalibrasi dan memisahkan diskriminasi dari kalibrasi.
6. **Tulis ketiga koefisien terfit apa adanya di artefak JSON.** Tiga angka
   yang bisa dilihat langsung adalah bukti paling meyakinkan bahwa ini bukan
   model terlatih.
7. **Selalu berdampingan dengan arm `w` tetap** pada split yang sama.

**Gerbang G6.3:** pada sweep validasi, A3 > max(A1, A2) dengan margin > 1 std
antar-seed.

#### F6-3b — Fusi adaptif **nol-parameter** (~0.5 hari, tidak menunggu keputusan)
Tingkat 2 pada taksonomi §2. Bobot bergantung trial tapi **tidak ada satu pun
parameter yang di-*fit***:

- bobot per-query dari margin z-score tiap ruang (selisih top-1 vs top-2 —
  ruang yang lebih "yakin" pada query itu dapat bobot lebih besar);
- alternatif: fusi berbasis peringkat (rank fusion), bukan berbasis skor.

**Kenapa ini dikerjakan lebih dulu:** murah, tidak butuh keputusan pembimbing,
dan menjadi jawaban kalau F6-3 ditolak. **Kejujuran yang harus ditulis di
naskah:** literatur SV hampir seluruhnya *men-fit* bobot fusinya (termasuk
quality-aware calibration Thienpondt dkk., ICASSP 2021), jadi F6-3b dilaporkan
sebagai **eksplorasi**, bukan sebagai penerapan resep yang sudah mapan.

**Gerbang G6.3b:** sama dengan G6.3.

**Pre-registrasi pelaporan (berlaku untuk F6-3 dan F6-3b):** laporkan **tiga
arm fusi** — `w` tetap / adaptif nol-parameter / LLR — pada protokol yang
sama. Ini mengisolasi kontribusi tiap tingkat taksonomi, dan tabel itu sendiri
adalah kontribusi metodologis tesis, apa pun hasil angkanya.

#### F6-4 — Aturan deteksi dua ruang (~1 hari)
Uji ulang temuan G4.3 (`mean`) pada 530 query unknown paruh-deteksi (**C6**,
**C13**), bersama aturan LLR dari F6-3.

**Gerbang G6.4:** EER A3 < min(EER A1, EER A2), CI bootstrap tidak memotong nol.

#### F6-5 — Run resmi (~1 hari)
`scripts/run_full_evaluation.py` dengan tag `exp6_*`, 10 repetisi, ablasi penuh
A1/A2/A3/B1 + 3 baseline, paired t-test. Bandingkan langsung dengan exp5b.

#### F6-6 — ❌ DICORET: backend terlatih
Head gaya ECAPA/TDNN di atas fitur multi-layer Whisper, dilatih AAM-softmax
dengan LoRA pada encoder — resep Whisper-PMFA penuh. **Melanggar batasan
bebas-pelatihan** (lihat §2 pembuka); `docs/experiment-5.md` §78 sudah memakai
batasan itu untuk menolak justru jalur ini. Selain itu `base_train` hanya
punya 111 speaker ber-cache vs VoxCeleb2 5994 yang dipakai Whisper-PMFA, jadi
overfit hampir pasti. **Tidak dikerjakan** kecuali pembimbing melonggarkan
batasan secara eksplisit.

---

## 3. Ringkasan usaha & risiko

| Fase | Effort | Risiko | Bisa dibuang? |
|---|---|---|---|
| F6-0 kebersihan + exp4 v2 | ~1 hari | rendah | **tidak** — memperbaiki C0/C1/C5/C6/C7/C9/C10 |
| F6-1 readout | 1–2 hari | **sedang-tinggi** | tidak — inti hipotesis |
| F6-2 ceiling | 3 jam | rendah | tidak |
| F6-3 LLR fusion | 1 hari | sedang | tidak — inti kontribusi (**menunggu keputusan pembimbing**) |
| **F6-3b fusi adaptif nol-parameter** | **0.5 hari** | **rendah** | **tidak — satu-satunya jalur yang tidak bergantung keputusan siapa pun** |
| F6-4 deteksi | 1 hari | rendah | ya, kalau waktu mepet |
| F6-5 run resmi | 1 hari | rendah | tidak |
| ~~F6-6 backend~~ | — | — | **dicoret — melanggar batasan bebas-pelatihan** |

**Total jalur utama: ~5.5 hari kerja. Nol training gradien** — seluruh jalur
utama adalah aritmatika di atas backbone beku plus statistik yang di-fit di
`base_train`, kecuali 3 koefisien LLR di F6-3 yang menunggu keputusan.
F6-3b tidak men-*fit* apa pun, jadi ia berjalan tanpa syarat.

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

Status venue diverifikasi 2026-08-30. **Klaim inti harus bersandar pada baris
ber-peer-review**; pracetak boleh sebagai pelengkap, tidak sebagai tumpuan.

### Ber-peer-review — aman dikutip sebagai dasar

| Rujukan | Venue | Dipakai untuk |
|---|---|---|
| Zhao dkk., [*Whisper-PMFA*](https://arxiv.org/abs/2408.15585) | **Interspeech 2024** | dasar F6-1. Klaim terverifikasi: EER **1.42 %** VoxCeleb1, **0.58 % absolut** di bawah ECAPA-TDNN; LoRA menurunkan parameter terlatih ~45× (— jalur LoRA itu justru yang dicoret di F6-6) |
| Okabe dkk., [*Attentive Statistics Pooling*](https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html) | Interspeech 2018 | pooling mean **+ std** di F6-1 |
| Chen dkk., [*WavLM*](https://arxiv.org/abs/2110.13900) | *IEEE JSTSP* 16(6), 2022 | weighted-sum antar layer |
| Lee & Lee, [*MMFA*](https://doi.org/10.3390/electronics14193857) | *Electronics* 14(19):3857, 2025 | agregasi multi-layer + masking frame |
| Mu & Viswanath, [*All-but-the-Top*](https://openreview.net/forum?id=HkuGJ3kCb) | **ICLR 2018** | ABTT — **default post-processing F6-1** (C3) |
| Huang dkk., [*WhiteningBERT*](https://aclanthology.org/2021.findings-emnlp.23/) | Findings of EMNLP 2021 | PCA-whitening (menggantikan pracetak `arXiv:2103.15316` yang dikutip draf lama) |
| Hatch, Kajarekar & Stolcke, [*WCCN*](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html) | Interspeech 2006 | WCCN di tabel §pengondisian |
| Snyder dkk., [*X-Vectors*](https://dblp.org/rec/conf/icassp/SnyderGSPK18.html) | ICASSP 2018 | reduksi LDA sebelum scoring (ablasi F6-1) |
| Garcia-Romero & Espy-Wilson, [*i-vector length normalization*](https://www.isca-archive.org/interspeech_2011/garciaromero11_interspeech.html) | Interspeech 2011 | L2-norm tiap paruh sebelum fusi |
| Brümmer dkk., [*STBU fusion*](https://ieeexplore.ieee.org/document/4291590/) | **IEEE TASLP 15(7), 2007** | **jangkar peer-review F6-3** — fusi regresi logistik sebagai *calibration and fusion* |
| Morrison, [*Tutorial on logistic-regression calibration and fusion*](https://arxiv.org/abs/2104.08846) | *Australian Journal of Forensic Sciences* 45(2), 2013 | turunan `LLR = a₀ + Σ aᵢ·sᵢ` |
| Arevalo dkk., [*Gated Multimodal Units*](https://arxiv.org/abs/1702.01992) | ICLR-W 2017 / [*NCAA* 32, 2020](https://link.springer.com/article/10.1007/s00521-019-04559-1) | keluarga `GatedAttentionFusion` — jalur yang **ditinggalkan** |
| Thienpondt dkk., [*IDLab VoxSRC-20*](https://arxiv.org/abs/2010.11255) | ICASSP 2021 | quality-aware score calibration — preseden bahwa literatur SV *men-fit* bobot fusi |

> **Koreksi sitasi 2026-08-30.** Draf sebelumnya mengatribusikan tutorial
> `arXiv:2104.08846` kepada **Brümmer**. Penulisnya **Geoffrey Stewart
> Morrison**; Brümmer hanya diberi *acknowledgement*. Ini rujukan utama F6-3 —
> penguji yang membuka tautannya akan langsung melihat selisihnya.

### Belum ber-peer-review — boleh disebut, jangan jadi tumpuan

- Brümmer & de Villiers, [*The BOSARIS Toolkit*](https://arxiv.org/abs/1304.2865)
  (2013) — **technical report**. Implementasi rujukan LLR fusion; selalu
  pasangkan dengan Brümmer dkk. 2007 di atas.
- [*AMECxSV*](https://arxiv.org/abs/2607.16532) (pracetak arXiv, Juli 2026) —
  quality-aware calibration; relevan untuk `a₃` di F6-3, tapi belum diuji
  sejawat.
- Zhou & Xie, [*Feature Alignment Determines Fusion Strategy*](https://arxiv.org/abs/2606.01207)
  (pracetak arXiv, Mei 2026) — dipakai di re-audit untuk mendukung pilihan
  konkatenasi. **Sandaran terlalu tipis untuk keputusan desain**; untungnya
  keputusan itu berdiri di atas bukti internal (galat 2.6e-06, §2 re-audit),
  bukan di atas makalah ini.

# Requirements

Konvensi ID: `FR-xx` (functional), `NFR-xx` (non-functional), `DR-xx` (data),
`ER-xx` (environment/tooling). Rujukan `[x.y]` mengarah ke sub-bab proposal,
`[Pers. x.y]` ke nomor persamaan.

## 1. Functional Requirements

### 1.1 Preprocessing Audio `[4.4]`

- **FR-01**: Sistem harus melakukan resampling seluruh audio input ke 16 kHz.
- **FR-02**: Sistem harus menormalisasi loudness ke target **-20 dB LUFS**.
- **FR-03**: Sistem harus menerapkan reduksi noise dasar sebelum tahap VAD.
- **FR-04**: Sistem harus mendeteksi segmen ucapan menggunakan **Silero VAD**
  dan membuang segmen non-ucapan (silence/noise murni).
- **FR-05**: Sistem harus menggabungkan (aggregate) segmen ucapan terdeteksi
  menjadi satu berkas audio bersih per utterance.
- **FR-06**: Standardisasi durasi via **padding tanpa truncation**:
  - Training ECAPA-TDNN: padding sirkular ke 3 detik untuk audio < 3s.
  - Training/inferensi Whisper: padding silence ke 30 detik.
  - Inferensi/enrollment: durasi penuh dipakai; untuk audio sangat panjang,
    gunakan *sliding window* + rata-rata embedding antar-window.
- **Acceptance**: tidak ada audio yang informasinya terpotong (truncated) di
  tahap mana pun kecuali di dalam sliding-window aggregation itu sendiri.

### 1.2 Ekstraksi Fitur Spektral `[3.2, 4.5]`

- **FR-07**: Sistem harus menghasilkan Log-Mel Spectrogram dengan parameter
  tetap: sample rate 16 kHz, 80 filter Mel, window 25 ms (400 sampel), hop
  10 ms (160 sampel); window function Hamming untuk jalur ECAPA-TDNN dan Hann
  untuk jalur Whisper.
- **Acceptance**: satu pipeline Log-Mel bersama dapat dipakai kedua backbone
  (parameter identik kecuali window function).

### 1.3 Ekstraksi Speaker Embedding (Multi-Backbone) `[3.3, 3.4, 4.5]`

- **FR-08**: Sistem harus mengekstraksi embedding dari **ECAPA-TDNN**
  (dimensi 192) menggunakan attentive statistics pooling bawaan arsitektur.
- **FR-09**: Sistem harus mengekstraksi embedding dari **Whisper encoder**
  (dimensi 512–1280 tergantung ukuran model) menggunakan lapisan pooling
  tambahan pada output encoder (bukan decoder).
- **FR-10**: Kedua backbone dapat memakai pretrained weights (tidak wajib
  dilatih dari nol) — ECAPA-TDNN pretrained speaker verification, Whisper
  pretrained ASR multilingual.
- **Acceptance**: masing-masing backbone menghasilkan satu vektor embedding
  berdimensi tetap per utterance (utterance-level, bukan frame-level).

### 1.4 Embedding Fusion `[3.5, Pers. 4.1-4.3]`

- **FR-11**: Sistem harus memproyeksikan kedua embedding backbone ke ruang
  dimensi bersama (256-d) via lapisan linear terpisah per backbone
  (Persamaan 4.1).
- **FR-12**: Sistem harus menghitung *gate* g via sigmoid dari concatenation
  kedua embedding terproyeksi (Persamaan 4.2).
- **FR-13**: Sistem harus menggabungkan kedua embedding terproyeksi secara
  element-wise weighted memakai gate g (Persamaan 4.3): `e_fusion = g ⊙ e'_1 + (1-g) ⊙ e'_2`.
- **FR-14**: Bobot fusion (W_i, b_i, W_g, b_g) harus **trainable**, dilatih
  end-to-end bersama seluruh jaringan — **bukan** bobot statis/manual.
- **Acceptance**: mengganti urutan backbone (ECAPA lalu Whisper vs sebaliknya)
  tidak mengubah hasil (fusion simetris secara desain, hanya beda peran e'_1/e'_2).

### 1.5 Prototypical Network & Klasifikasi `[3.6, 3.7, 4.6]`

- **FR-15**: Sistem harus mendukung episodic sampling N-way K-shot untuk
  training (support set + query set per episode).
- **FR-16**: Sistem harus menghitung prototipe kelas sebagai rata-rata
  embedding support set (Persamaan 3.6/3.7).
- **FR-17**: Sistem harus mengklasifikasi query berdasarkan jarak Euclidean
  ke seluruh prototipe, dikonversi ke probabilitas via softmax jarak-negatif
  (Persamaan 3.8-3.9).
- **FR-18**: Sistem harus dilatih dengan negative log-likelihood loss
  (Persamaan 3.10).

### 1.6 Open-Set Identification & Threshold `[3.8, 4.6 poin 3-4]`

- **FR-19**: Sistem harus membandingkan jarak minimum query-ke-prototipe
  terhadap ambang batas (threshold); di bawah threshold → known speaker,
  di atas → unknown speaker.
- **FR-20**: Threshold harus ditentukan via kalibrasi terpisah: kurva
  ROC/DET dari pasangan **genuine** (sample vs prototipe pembicara sama, dari
  Data Latih Awal) dan **impostor** (sample vs prototipe pembicara berbeda,
  dari sisa Data Uji Global di luar 100 pembicara sesi inkremental & data
  simpanan 10%), diambil pada titik **EER**.
- **FR-21**: Threshold hasil kalibrasi bersifat **fixed** — dipakai konsisten
  di seluruh sesi Level Task/Episodic Split, **tidak** dikalibrasi ulang per
  sesi.
- **Acceptance**: data yang dipakai kalibrasi threshold harus terbukti
  disjoint dari data sesi inkremental & data simpanan (cek otomatis).

### 1.7 Continual Learning — Prototype Update `[3.8, Pers. 4.4-4.6]`

- **FR-22**: Sistem harus memperbarui prototipe pembicara lama via
  **running average tertimbang** (bukan exponential moving average):
  `n_k(t) = n_k(t-1) + m`, `c_k(t) = [n_k(t-1)·c_k(t-1) + m·z̄_new] / n_k(t)`.
- **FR-23**: Sistem harus menyimpan hanya pasangan `(c_k, n_k)` per
  pembicara — **tidak** menyimpan embedding mentah historis.
- **FR-24**: Inisialisasi prototipe pembicara baru memakai rata-rata
  embedding support set pertama (Persamaan 4.4), konsisten dengan
  Persamaan 4.5-4.6.
- **FR-25**: Sistem harus mendukung update ini **tanpa retraining ulang**
  seluruh model (backbone + fusion tetap beku saat prototype update).

### 1.8 Deteksi & Registrasi Pembicara Baru (Novel Speaker) `[4.7]`

- **FR-26**: Sampel yang terdeteksi unknown (FR-19) harus diakumulasi dalam
  buffer sementara — **tidak** langsung didaftarkan sebagai prototipe baru
  dari satu sampel tunggal.
- **FR-27**: Registrasi prototipe baru butuh **dua syarat sekaligus**:
  (a) jumlah sampel di buffer mencapai minimum sesuai skema one/few-shot,
  (b) rata-rata Silhouette Coefficient kelompok kandidat (Persamaan 4.7)
  mendekati +1 (kompak & terpisah dari prototipe lama).
- **FR-28**: Jika syarat belum terpenuhi, sistem menunggu akumulasi sampel
  tambahan — tidak memaksakan keputusan.

### 1.9 Speaker Database `[3.8, 4.1]`

- **FR-29**: Prototipe yang diperbarui/baru harus disimpan ke speaker
  database sehingga identifikasi berikutnya memakai representasi terbaru.

### 1.10 Evaluasi & Pelaporan Metrik `[3.9, 4.8]`

- **FR-30**: Sistem harus menghitung **Accuracy** (Persamaan 3.11).
- **FR-31**: Sistem harus menghitung **EER** dari FAR/FRR di seluruh
  threshold (Persamaan 3.12-3.13).
- **FR-32**: Sistem harus menghitung **Average Accuracy** lintas T tugas
  (Persamaan 3.14).
- **FR-33**: Sistem harus menghitung **Forgetting Measure** (Persamaan 3.15).
- **FR-34**: Harness evaluasi harus menjalankan protokol **FSCIL 10 sesi
  inkremental × 10-way, K=1** (adaptasi Tao et al. 2020), dengan re-test
  gabungan pembicara lama+baru setiap sesi.

### 1.11 Ablation Study `[4.9]`

- **FR-35**: Sistem harus bisa dikonfigurasi ke 3 mode backbone: ECAPA-only
  (A1), Whisper-only (A2), fusion lengkap (A3) — komponen lain identik.
- **FR-36**: Sistem harus bisa dikonfigurasi ke 2 mode prototipe: statis/beku
  setelah base training (B1), running-average update (B2) — komponen lain
  identik.

### 1.12 Baseline Comparison `[4.10]`

- **FR-37**: Sistem harus menyediakan implementasi/re-run baseline pada
  protokol identik: x-vector+PLDA, ECAPA-TDNN standar (tanpa
  one-shot/open-set/continual), Prototypical Network vanilla (tanpa
  open-set & continual), openFEAT (atau versi linear-adaptor sederhana jika
  Transformer set-to-set penuh tidak feasible — nyatakan penyederhanaan
  secara eksplisit).

### 1.13 Uji Signifikansi Statistik `[4.11]`

- **FR-38**: Setiap konfigurasi ablation/baseline harus dijalankan **10
  pengulangan independen** (seed berbeda untuk episodic sampling), hasil
  dilaporkan mean ± std.
- **FR-39**: Uji normalitas (Shapiro-Wilk) → paired t-test jika normal,
  Wilcoxon signed-rank jika tidak, untuk metrik Accuracy.
- **FR-40**: Bootstrap resampling untuk interval kepercayaan 95% selisih EER
  (metodologi Bengio & Mariéthoz, 2004).
- **FR-41**: Koreksi Bonferroni pada ambang signifikansi α=0.05 dibagi jumlah
  total perbandingan berpasangan.

## 2. Non-Functional Requirements

- **NFR-01 (Reproducibility)**: seluruh eksperimen harus reproducible —
  fixed random seed per run, versi pretrained model & dataset dicatat,
  environment terkunci (lockfile/requirements pinned).
- **NFR-02 (No data leakage)**: speaker-disjoint split harus diverifikasi
  otomatis (unit test) di setiap level split (base split, task split,
  kalibrasi threshold).
- **NFR-03 (Traceability)**: setiap komponen kode harus dapat ditelusuri ke
  sub-bab/persamaan proposal terkait (komentar/README per modul).
- **NFR-04 (Compute efficiency)**: embedding backbone yang beku (frozen)
  harus di-cache agar tidak dihitung ulang pada setiap pengulangan
  eksperimen ablation/statistik.
- **NFR-05 (Modularitas)**: preprocessing, backbone, fusion, prototypical
  network, dan continual learning harus jadi modul terpisah yang bisa
  diuji/diganti independen (mendukung ablation study FR-35/FR-36).
- **NFR-06 (Auditability metrik)**: perhitungan EER/threshold harus bisa
  diinspeksi via kurva ROC/DET yang disimpan (plot + raw FAR/FRR per
  threshold), bukan hanya angka akhir.
- **NFR-07 (Lisensi & etika data)**: penggunaan VoxCeleb1/2 harus mengikuti
  lisensi CC BY 4.0 dan mencantumkan atribusi sesuai `[4.3]`.

## 3. Data Requirements `[4.3]`

- **DR-01**: Dataset: VoxCeleb1 (1.251 speaker, 153.516 utterance, ≈352 jam)
  + VoxCeleb2 (6.112 speaker, 1.128.246 utterance, ≈2.442 jam), keduanya
  speaker-disjoint, format WAV mono 16-bit PCM.
- **DR-02**: Split level global: 70% Data Latih Awal (≈5.154 speaker) / 30%
  Data Uji Global (≈2.209 speaker).
- **DR-03**: Dari Data Uji Global: 10% (≈221 speaker) jadi data simpanan
  (pool unknown tambahan untuk open-set), sisanya ≈1.988 speaker untuk
  Level Task/Episodic Split.
- **DR-04**: Skema task: 10 sesi inkremental × 10-way × K=1 → butuh 100
  speaker baru total (≈4,5% dari 2.209 — jauh di bawah kapasitas).
- **DR-05**: Data kalibrasi threshold (FR-20) harus disusun dari pasangan
  genuine (Data Latih Awal) dan impostor (sisa Data Uji Global di luar 100
  speaker sesi + data simpanan) — **terpisah tegas** dari data pengujian.
- **DR-06**: Support set per task: 1–5 sampel/speaker; query set: campuran
  speaker lama+baru, dipisah acak-terkontrol.

## 4. Environment / Tooling Requirements

- **ER-01**: Python 3.10+, PyTorch (GPU/CUDA) sebagai framework utama.
- **ER-02**: Library backbone: SpeechBrain atau implementasi ECAPA-TDNN
  pretrained (VoxCeleb-trained), `openai-whisper` atau HuggingFace
  `transformers` untuk Whisper encoder.
- **ER-03**: Silero VAD (`torch.hub` / `silero-vad` package).
- **ER-04**: Audio I/O & DSP: `torchaudio` atau `librosa` untuk
  resampling, Log-Mel extraction, loudness normalization (LUFS via
  `pyloudnorm`).
- **ER-05**: Statistik: `scipy.stats` (Shapiro-Wilk, t-test, Wilcoxon),
  bootstrap custom atau `scikit-learn`/`scipy` resampling utilities.
- **ER-06**: Clustering/validasi: `sklearn.metrics.silhouette_score` untuk
  Silhouette Coefficient (FR-27).
- **ER-07**: Eksperimen tracking: mekanisme pencatatan hasil per-run
  (mis. CSV/JSON terstruktur atau MLflow/W&B) agar mean±std FR-38 mudah
  diagregasi.
- **ER-08**: GPU dengan VRAM cukup untuk Whisper encoder (varian medium/
  large) + ECAPA-TDNN secara bersamaan saat training fusion; jika terbatas,
  precompute & cache embedding backbone terlebih dahulu (lihat NFR-04).
- **ER-09**: Storage: ruang disk untuk VoxCeleb1+2 (~2.8k jam audio) dan
  cache embedding.

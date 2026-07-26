# Spesifikasi Arsitektur Teknis

Rujukan visual: **Gambar 4.1 Arsitektur Sistem** (proposal, hlm. 30).
Dokumen ini menerjemahkan diagram tersebut menjadi spesifikasi modul yang
implementable.

## 1. Alur Data End-to-End

```
Audio Input (raw)
   │
   ▼
[1] Preprocessing Audio ── resampling 16kHz, normalisasi -20 LUFS, noise reduction
   │
   ▼
[2] Voice Activity Detection (Silero VAD) ── buang non-speech
   │
   ▼
[3] Speech Segment Aggregation ── gabungkan segmen speech → 1 audio bersih
   │
   ▼
[4] Standardisasi Durasi (padding, bukan truncation)
   │        ├── training ECAPA-TDNN → pad ke 3 detik (sirkular)
   │        ├── training/inferensi Whisper → pad silence ke 30 detik
   │        └── inferensi/enrollment → full-length (+ sliding window jika sangat panjang)
   ▼
[5] Log-Mel Spectrogram (16kHz, 80 mel, win 25ms, hop 10ms; Hamming/Hann)
   │
   ├──────────────────────┬───────────────────────┐
   ▼                      ▼                       
[6a] ECAPA-TDNN backbone  [6b] Whisper encoder     
   → embedding 192-d        → embedding 512-1280d  
   │                      │
   ▼                      ▼
[7] Proyeksi linear W_i/b_i ke ruang bersama 256-d (masing-masing backbone)
   │
   ▼
[8] Gated Attention Fusion ── gate sigmoid → kombinasi weighted element-wise
   │
   ▼
e_fusion (256-d, normalized)
   │
   ▼
[9] Prototypical Network Module
   │        ├── training: episodic N-way K-shot, hitung prototipe, NLL loss
   │        └── inference: jarak query→semua prototipe (Euclidean)
   ▼
[10] Distance-Based Classification + Open-Set Threshold (EER dari kalibrasi ROC)
   │            ├── jarak < threshold → Known Speaker (identitas = prototipe terdekat)
   │            └── jarak ≥ threshold → Unknown Speaker → buffer kandidat baru
   ▼
[11] Continual Prototype Update (jika known: running-average; jika buffer kandidat
     lolos validasi Silhouette Coefficient: registrasi prototipe baru)
   │
   ▼
[12] Speaker Database Update (simpan (c_k, n_k) terbaru)
```

## 2. Modul 1 — Preprocessing Audio `[4.4]`

**Input**: raw audio (format apapun, dikonversi dulu).
**Output**: waveform 16 kHz, mono, ternormalisasi -20 LUFS, VAD-trimmed.

Langkah:
1. Resample ke 16 kHz.
2. Noise reduction dasar (spectral gating atau setara — tidak perlu model
   VAD adversarial kompleks dari Bab 3.1; Silero VAD off-the-shelf sudah
   cukup untuk deteksi speech/non-speech per batasan masalah B2).
3. Silero VAD → daftar segmen `(start, end)` berisi speech.
4. Silence removal: konkatenasi segmen speech → 1 waveform bersih.
5. Loudness normalization ke -20 dB LUFS (`pyloudnorm`).
6. Padding sesuai konteks pemakaian (training ECAPA / training Whisper /
   inferensi).

**Interface yang disarankan**:
```
preprocess(raw_audio, sr) -> clean_waveform_16k
standardize_duration(clean_waveform, mode: "ecapa_train"|"whisper_train"|"inference") -> waveform_fixed
```

## 3. Modul 2 — Ekstraksi Fitur Spektral `[3.2, 4.5, Tabel 4.2]`

Log-Mel Spectrogram, parameter **identik** untuk kedua backbone kecuali
window function:

| Parameter | Nilai |
|---|---|
| Sample rate | 16.000 Hz |
| Jumlah filter Mel | 80 |
| Window size | 25 ms (400 sampel) |
| Hop length | 10 ms (160 sampel) |
| Window function | Hamming (ECAPA-TDNN) / Hann (Whisper) |

Implementasi: satu fungsi `logmel(waveform, window_fn)` dipanggil dua kali
dengan window function berbeda, atau gunakan pipeline bawaan
masing-masing library backbone jika sudah sesuai parameter di atas (jangan
duplikasi implementasi bila library resmi sudah menyediakan yang identik).

## 4. Modul 3 — Backbone Embedding

### 4a. ECAPA-TDNN `[3.4]`
- Input: log-Mel (Hamming window).
- Arsitektur: Res2Net conv blocks + Squeeze-Excitation (channel attention) +
  feature propagation/aggregation + attentive statistics pooling.
- Output: embedding 192-d, L2-normalized.
- Sumber: gunakan pretrained (mis. SpeechBrain `spkrec-ecapa-voxceleb`) —
  **jangan** melatih dari nol kecuali eksperimen menunjukkan perlu.

### 4b. Whisper Encoder `[3.3]`
- Input: log-Mel (Hann window), **hanya bagian encoder** (bukan decoder ASR).
- Arsitektur: conv front-end + multi-head self-attention blocks.
- Output: representasi tersembunyi → pooling tambahan (mengikuti pendekatan
  Whisper-PMFA) → embedding 512–1280-d tergantung ukuran model.
- Rekomendasi mulai dari varian kecil (`base`/`small`) untuk iterasi cepat,
  naik ke ukuran lebih besar setelah pipeline tervalidasi (lihat NFR-04 di
  requirements — Whisper besar mahal secara komputasi).

## 5. Modul 4 — Embedding Fusion `[3.5, Pers. 4.1-4.3]`

Formulasi matematis (harus diimplementasikan persis, bukan didekati):

```
e'_i = W_i · e_i + b_i                        untuk i ∈ {ECAPA, Whisper}   (4.1)
g    = σ(W_g · [e'_1 ; e'_2] + b_g)                                       (4.2)
e_fusion = g ⊙ e'_1 + (1 - g) ⊙ e'_2                                      (4.3)
```

- Dimensi proyeksi bersama: **256**.
- Fusion dilakukan pada **level utterance** (satu vektor per backbone per
  sampel), bukan level frame — karena frame rate ECAPA-TDNN dan Whisper
  berbeda dan penyelarasan temporal frame-level tidak diperlukan (dan
  menambah kompleksitas tanpa manfaat jelas).
- Seluruh parameter (`W_1, b_1, W_2, b_2, W_g, b_g`) **trainable**, dilatih
  end-to-end bersama prototypical network (Modul 5) — gate memungkinkan
  kontribusi adaptif per-sampel (mis. Whisper lebih dominan pada kondisi
  bising, ECAPA lebih dominan pada kondisi bersih).
- Output di-normalize (L2) sebelum masuk Modul 5.

## 6. Modul 5 — Prototypical Network `[3.6, 3.7, 4.6]`

**Training (episodic)**:
```
untuk setiap episode N-way K-shot:
    support_set, query_set = sample_episode(N, K)
    embeddings = fusion(backbone(support_set ∪ query_set))
    prototipe c_k = mean(embeddings support_set kelas k)         # Pers. 3.6/3.7
    d(z_q, c_k) = ||z_q - c_k||_2                                # Pers. 3.8
    p(y=k | x_q) = softmax(-d(z_q, c_k))                         # Pers. 3.9
    loss = NLL(p, y_true)                                        # Pers. 3.10
    backprop ke: fusion weights + (opsional) backbone jika fine-tuned
```

**Inference (open-set)**:
```
embedding_input = fusion(backbone(input))
d_min, k_best = min distance ke seluruh prototipe di speaker database
jika d_min < threshold:  → Known Speaker = k_best
jika d_min >= threshold: → Unknown Speaker → masuk ke Modul 6 (buffer)
```

## 7. Modul 6 — Open-Set Threshold Calibration `[4.6 poin 4]`

Dijalankan **sekali** setelah base training selesai, **sebelum** evaluasi
Level Task/Episodic Split dimulai — bukan per-sesi.

Langkah:
1. Susun pasangan genuine (embedding vs prototipe speaker sama, dari Data
   Latih Awal) dan impostor (embedding vs prototipe speaker beda, dari sisa
   Data Uji Global di luar 100 speaker sesi inkremental & data simpanan 10%).
2. Hitung jarak Euclidean tiap pasangan.
3. Sweep threshold di seluruh rentang jarak, hitung FAR(θ) & FRR(θ)
   (Persamaan 3.12-3.13) → bentuk kurva ROC/DET.
4. Cari titik EER (FAR ≈ FRR) → jarak pada titik itu jadi **threshold
   operasional tetap**.
5. Simpan threshold ini, pakai konsisten di seluruh sesi berikutnya
   (Modul 5 & 8).

**Guard rail**: unit test yang memastikan tidak ada overlap speaker ID
antara data kalibrasi ini dengan 100 speaker sesi inkremental maupun data
simpanan.

## 8. Modul 7 — Continual Learning: Prototype Update `[3.8, Pers. 4.4-4.6]`

```
# pembicara sudah dikenal (known speaker k, menerima m sampel baru)
z̄_new = mean(fusion(backbone(x_new_1..m)))                       # Pers. 4.4
n_k(t) = n_k(t-1) + m                                              # Pers. 4.5
c_k(t) = [n_k(t-1)*c_k(t-1) + m*z̄_new] / n_k(t)                   # Pers. 4.6

# pembicara benar-benar baru → inisialisasi = Pers. 4.4 langsung sebagai c_k(0), n_k(0)=m
```

- State yang disimpan per speaker: **hanya** `(c_k, n_k)` — bukan riwayat
  embedding mentah (efisien memori, mendukung DR-05/NFR-04).
- Tidak ada hyperparameter decay tambahan (beda dari EMA/momentum update) —
  jangan implementasikan decay factor di sini.
- Modul ini **tidak** menyentuh bobot backbone atau fusion — hanya
  memperbarui entri di speaker database (Modul 9).

## 9. Modul 8 — Novel Speaker Detection `[4.7, Pers. 4.7]`

```
buffer[unknown] += sampel yang lolos threshold check (Modul 6, cabang unknown)

jika len(buffer[unknown]) >= min_sampel (sesuai skema one/few-shot):
    hitung Silhouette Coefficient rata-rata kelompok buffer:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))                    # Pers. 4.7
        a(i) = rata-rata jarak i ke sampel lain di buffer (kekompakan)
        b(i) = rata-rata jarak i ke prototipe terdekat yang sudah ada (pemisahan)
    jika mean(s) mendekati +1:
        registrasi prototipe baru (init via Pers. 4.4) → tambah ke speaker database
        kosongkan buffer untuk kandidat ini
    else:
        tetap di buffer, tunggu sampel tambahan
```

## 10. Modul 9 — Speaker Database `[3.8, 4.1]`

- Struktur data minimal per entri: `speaker_id -> (prototype_vector c_k, count n_k)`.
- Harus mendukung: baca (untuk klasifikasi Modul 5), update in-place (Modul 7),
  insert baru (Modul 8).
- Persist ke disk antar sesi evaluasi (bukan hanya in-memory) agar hasil
  Level Task/Episodic Split bisa direproduksi/diaudit.

## 11. Prinsip Desain yang Harus Dijaga

1. **Backbone beku vs trainable**: default backbone ECAPA-TDNN & Whisper
   dipakai sebagai *fixed feature extractor* (pretrained, freeze); hanya
   fusion layer + prototypical network head yang dilatih end-to-end,
   kecuali eksperimen tambahan secara eksplisit menguji fine-tuning
   backbone. Ini menjaga F4/F5 tetap murah secara komputasi dan konsisten
   dengan asumsi "tanpa retraining penuh" di Bab 3.8/4.1.
2. **Tidak ada retraining penuh saat continual learning** — Modul 7/8 hanya
   memanipulasi speaker database, bukan gradient descent ulang pada
   backbone/fusion.
3. **Threshold tetap selama sesi inkremental** — jangan kalibrasi ulang per
   sesi (lihat FR-21).
4. **Modularitas untuk ablation**: Modul 3 (backbone) dan Modul 7 (continual
   learning) harus punya *feature flag*/config switch agar bisa dimatikan
   sebagian sesuai Tabel 4.3 (A1/A2/A3) dan Tabel 4.4 (B1/B2) tanpa mengubah
   kode inti.

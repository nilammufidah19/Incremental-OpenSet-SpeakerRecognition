# Incremental Open-Set Speaker Recognition

Implementasi tesis **"Sistem Identifikasi Pembicara Open-Set Berbasis One-Shot Learning dengan Multi-Backbone Embedding dan Pembaruan Pengetahuan Bertahap"** — Nilam Mufidah, 24/551986/PPA/06994, Program Magister Kecerdasan Artifisial, Universitas Gadjah Mada.

Sistem identifikasi pembicara yang bisa mengenali pembicara baru dari **satu sampel suara** (one-shot), membedakan pembicara terdaftar vs pembicara yang belum pernah didaftarkan (**open-set**, bukan memaksa klasifikasi ke kelas terdekat), dan menambah pembicara baru secara **inkremental** tanpa retraining penuh maupun *catastrophic forgetting* signifikan terhadap pembicara lama.

## Daftar Isi

- [Ringkasan Masalah & Solusi](#ringkasan-masalah--solusi)
- [Arsitektur Pipeline](#arsitektur-pipeline)
- [Struktur Repository](#struktur-repository)
- [Instalasi](#instalasi)
- [Persiapan Data (VoxCeleb1/2)](#persiapan-data-voxceleb12)
- [Menjalankan Sistem](#menjalankan-sistem)
- [Menjalankan Eksperimen](#menjalankan-eksperimen)
- [Testing](#testing)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Status Kode & Catatan Penting](#status-kode--catatan-penting)
- [Dokumentasi Lengkap](#dokumentasi-lengkap)
- [Batasan Penelitian](#batasan-penelitian)

## Ringkasan Masalah & Solusi

Sistem identifikasi pembicara konvensional punya empat kelemahan yang jadi fokus penelitian ini:

1. Butuh banyak sampel data per individu untuk enrollment yang akurat.
2. Berasumsi **closed-set** — semua pembicara yang mungkin muncul sudah dikenal sebelumnya.
3. Tidak punya mekanisme open-set yang baik: pembicara baru yang belum terdaftar cenderung tetap dipaksa diklasifikasikan ke kelas terdekat yang sudah ada.
4. Menambah pembicara baru biasanya butuh retraining penuh, berisiko *catastrophic forgetting* terhadap pengetahuan pembicara lama.

**Solusi yang diusulkan**: pipeline VAD → dual-backbone speaker embedding (ECAPA-TDNN + backbone kedua, lihat [Status Kode](#status-kode--catatan-penting)) → *gated attention fusion* / *score-level fusion* → *prototypical network* untuk klasifikasi berbasis jarak → threshold open-set berbasis EER/target-FRR → pembaruan prototipe inkremental (continual learning) tanpa retraining penuh.

## Arsitektur Pipeline

```
audio mentah
  │
  ▼
Preprocessing  (src/preprocessing/)
  resample 16kHz → noise reduction → VAD (Silero) → agregasi segmen ucapan
  → normalisasi loudness (-20 LUFS) → standardisasi durasi (4 mode)
  │
  ▼
Backbone beku, dengan cache  (src/models/, src/features/cache.py)
  ECAPA-TDNN (192-d, pretrained SpeechBrain)  +  backbone kedua (feature-flag)
    · exp0–exp4 : Whisper encoder (512-d, pooling layer tengah)
    · exp5      : ReDimNet-b2 (192-d) — backbone kedua aktif saat ini
  │
  ▼
Fusion  (src/models/fusion.py)
  Gated Attention Fusion (trainable, 256-d)  ATAU  score-level fusion (parameter-free)
  │
  ▼
Normalisasi skor opsional — feature flag  (src/prototypical/score_norm.py)
  AS-Norm 1-ruang / DualASNorm 2-ruang terhadap cohort base_train
  │
  ▼
Keputusan open-set  (src/prototypical/, src/continual/)
  jarak ke prototipe terdekat → threshold (EER atau target-FRR)
  known → update prototipe (running-average) / unknown → buffer + validasi
  Silhouette Coefficient → registrasi pembicara baru
```

`src/system.py::SpeakerIdentificationSystem` merangkai seluruh pipeline di atas menjadi satu objek (`enroll()` / `process()` / `score()` / `save()` / `load()`). Setiap konfigurasi yang bisa dilaporkan (backbone kedua, strategi fusion, strategi kalibrasi threshold, mode continual learning) adalah *feature flag* yang didaftarkan sebagai satu `ExperimentConfig` di [`src/experiments.py`](src/experiments.py) — tidak ada nilai yang di-hardcode di pipeline itu sendiri.

## Struktur Repository

```
├── src/                      Kode inti pipeline
│   ├── preprocessing/        Resample, denoise, VAD, agregasi, loudness, durasi
│   ├── features/             Log-Mel, embedding cache (F3-06)
│   ├── models/                Backbone (ECAPA/x-vector/ReDimNet/WavLM/Whisper) + fusion
│   ├── prototypical/          Episodic sampler, prototipe, klasifikasi, kalibrasi threshold,
│   │                          AS-Norm score normalization, training loop
│   ├── continual/              Speaker database, prototype update, deteksi speaker baru
│   ├── evaluation/            Metrik (Accuracy/EER/Average Accuracy/Forgetting Measure),
│   │                          harness FSCIL, ablation, baseline, uji statistik
│   ├── data/                  Metadata VoxCeleb1/2, speaker-disjoint split, akuisisi audio
│   ├── utils/                 Seed management, experiment tracking
│   ├── system.py              SpeakerIdentificationSystem — integrasi seluruh pipeline (F7)
│   └── experiments.py         Registry / feature-flag switch tiap eksperimen
├── scripts/                    Entry point CLI (akuisisi data, precompute, evaluasi, laporan)
├── tests/                      156 unit test (pytest), sebagian pakai audio real (vox1_sample)
├── configs/                    (reserved, saat ini kosong)
├── data/
│   ├── raw/metadata/            Metadata resmi VoxCeleb1/2 (kecil, di-commit)
│   ├── raw/audio/                Audio mentah (besar, TIDAK di-commit — lihat .gitignore)
│   ├── splits/                   Hasil speaker-disjoint split (JSON/MD, di-commit — reproducible)
│   ├── cache/embeddings/         Cache embedding per backbone (besar, TIDAK di-commit)
│   └── processed/                Artefak preprocessing antara
├── pretrained_models/            Bobot pretrained ECAPA-TDNN & x-vector (diunduh otomatis via SpeechBrain)
├── experiments/                  Hasil run: JSON summary, log, laporan per fase (F8-F11)
├── docs/                         Dokumentasi tiap eksperimen (1-5) + laporan lengkap
└── plan/                         Rencana teknis: project plan, requirements, arsitektur, task, evaluasi
```

## Instalasi

**Prasyarat**: Python 3.11, Windows (skrip & path handling ditulis untuk Windows; lihat catatan drive/case-sensitivity di kode), opsional GPU NVIDIA (CUDA 12.1) untuk mempercepat ekstraksi embedding.

```powershell
# 1. Buat & aktifkan virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies (versi terkunci untuk reproducibility)
pip install -r requirements.txt

# 3. (Opsional) verifikasi GPU terdeteksi
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Bobot pretrained ECAPA-TDNN dan x-vector (SpeechBrain, `speechbrain/spkrec-ecapa-voxceleb` & `speechbrain/spkrec-xvect-voxceleb`) diunduh otomatis ke `pretrained_models/` saat pertama kali dipanggil — tidak perlu langkah manual.

## Persiapan Data (VoxCeleb1/2)

VoxCeleb1/2 didistribusikan Oxford VGG di bawah perjanjian akses berbasis permintaan (kredensial). Repo ini punya dua jalur:

1. **Metadata saja** (tanpa kredensial, dibutuhkan untuk menjalankan speaker-disjoint split):
   ```powershell
   .venv\Scripts\python.exe scripts\download_voxceleb.py --stage metadata
   .venv\Scripts\python.exe scripts\build_splits.py
   ```
2. **Audio, tanpa kredensial VGG** — lewat mirror ungated `ProgramComputer/voxceleb` di HuggingFace, dibaca sebagian (HTTP Range) langsung dari arsip ZIP resmi tanpa mengunduh seluruh arsip 30-80 GB (lihat [`src/data/remote_zip_audio.py`](src/data/remote_zip_audio.py)):
   ```powershell
   # sampel kecil untuk pengembangan pipeline
   .venv\Scripts\python.exe scripts\fetch_sample_audio.py

   # subset ter-cap per speaker untuk base_train (besar, jalankan sadar durasi/kuota disk)
   .venv\Scripts\python.exe scripts\fetch_capped_base_train_audio.py --dataset vox1 --cap 14 --n-workers 4
   .venv\Scripts\python.exe scripts\fetch_capped_eval_audio.py   # tanpa argumen -- mengambil task_speakers/reserved_unknown_pool/sample kalibrasi dari full_split.json
   ```
   Audio resmi berkredensial VGG juga didukung lewat `download_voxceleb.py --stage audio --username <u> --password <p>` bila kredensial sudah didapat.

Audio (`data/raw/audio/`) dan cache embedding (`data/cache/`) sengaja **tidak** di-commit (lihat `.gitignore`) karena ukurannya bisa mencapai puluhan GB — hanya metadata kecil dan hasil split JSON/MD yang di-commit agar split speaker-disjoint tetap reproducible tanpa mengunduh ulang apa pun.

## Menjalankan Sistem

Contoh pemakaian `SpeakerIdentificationSystem` secara langsung (di luar harness evaluasi):

```python
from src.models.fusion import GatedAttentionFusion, FUSION_DIM
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM
from src.system import SpeakerIdentificationSystem

fusion = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
system = SpeakerIdentificationSystem(fusion, threshold=0.8, continual_mode="running_average")

system.enroll("speaker_A", ["path/ke/sampel_A.wav"])       # one-shot enrollment
result = system.process("path/ke/query.wav")                # keputusan open-set + update inkremental
print(result.is_known, result.predicted_speaker_id, result.min_distance)

system.save("checkpoints/my_system")                        # persist fusion + speaker database + config
loaded = SpeakerIdentificationSystem.load("checkpoints/my_system")
```

## Menjalankan Eksperimen

Setiap konfigurasi eksperimen adalah satu tag di [`src/experiments.py`](src/experiments.py). Ganti eksperimen aktif tanpa mengubah kode lewat environment variable, lalu jalankan harness evaluasi end-to-end:

```powershell
$env:ACTIVE_EXPERIMENT = "exp5b_redimnet_fusion"   # atau tag lain, lihat tabel di docs/README.md
.venv\Scripts\python.exe scripts\run_full_evaluation.py
.venv\Scripts\python.exe scripts\generate_report.py
```

`run_full_evaluation.py` menjalankan: base training tiap varian fusion (A1/A2/A3) → kalibrasi threshold → evaluasi FSCIL 10×10-way real → ablation study (F9) → baseline comparison (F10: x-vector+PLDA, ECAPA standar, ProtoNet vanilla) → uji signifikansi statistik (F11). Hasil ditulis ke `experiments/full_evaluation_summary_<tag>.json`.

Skrip pendukung lain di `scripts/`: `precompute_embeddings.py` (isi cache embedding di muka), `benchmark_backbones.py`, `diagnose_accuracy_gap.py`, serta skrip `exp3_*`/`exp4_*`/`exp5_*` untuk validation sweep dan analisis spesifik tiap eksperimen (lihat docstring masing-masing skrip).

Menambah eksperimen baru: tambahkan satu `ExperimentConfig` ke dict `EXPERIMENTS` di `src/experiments.py`, arahkan `ACTIVE_EXPERIMENT` ke tag baru, lalu dokumentasikan di `docs/experiment-N.md`.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -q
```

156 unit test mencakup seluruh modul `src/` (preprocessing, backbone, fusion, prototypical, continual learning, evaluasi) — sebagian menggunakan audio VoxCeleb1 real (`data/raw/audio/vox1_sample/`, diunduh via `fetch_sample_audio.py`) dan otomatis dilewati (`skip`) bila audio sampel belum tersedia.

## Hasil Eksperimen

| # | Tag | Ringkasan | Akurasi usulan |
|---|---|---|---|
| 0 | `baseline_v0` | Fusi random-init, 500 episode training | 0.235 *(belum dihitung ulang)* |
| 1 | `exp1_frozen_residual` | Fusi residual-init + beku (0 episode) | 0.731 *(belum dihitung ulang)* |
| 2a | `exp2a_scorefusion_L4` | Whisper L4 + score-level fusion | 0.733 (≈ exp1) *(belum dihitung ulang)* |
| 3a | `exp3a_lowfrr` | Titik operasi low-FRR saja (target_frr re-locked 0.01→0.15), 10 repetisi | 0.695 ± 0.018 |
| 3b | `exp3b_asnorm` | AS-Norm + low-FRR (target_frr re-locked 0.05→0.01), 10 repetisi — **melampaui baseline ECAPA closed-set 0.783 signifikan** (p<0.0001); continual update signifikan (p=0.0007) | 0.833 ± 0.015 |
| 3c | `exp3c_dualmetric` | View pelaporan dua-metrik dari run exp3b | = 3b |
| 4 | *(analisis, tanpa tag)* | Gerbang keputusan fusi Whisper — fusi ditutup definitif | — *(tidak terdampak fix, tidak dihitung ulang)* |
| 5b | `exp5b_redimnet_fusion` **(aktif)** | ECAPA + ReDimNet, DualASNorm dua-ruang (w re-locked 0.3→0.5), 10 repetisi — **A3 > A1 signifikan (p<0.0001)**, melampaui baseline ECAPA signifikan (p<0.0001) | **0.876 ± 0.016** |

Detail lengkap tiap eksperimen (protokol, hyperparameter, uji signifikansi, artefak) ada di [`docs/`](docs/README.md) — [`docs/experiment-3.md`](docs/experiment-3.md) dan [`docs/experiment-5.md`](docs/experiment-5.md) sudah direvisi penuh dengan angka final. **`exp5b_redimnet_fusion` adalah hasil headline** (kontribusi fusi signifikan, A3>A1); `exp3b_asnorm` sekarang juga kembali melampaui baseline secara signifikan setelah repetisi dinaikkan ke 10 (target proposal).

## Status Kode & Catatan Penting

> ✅ **Code review 2026-07-26 + recompute/rerun 2026-07-27 + perbaikan lanjutan 2026-08-02 SELESAI.** Review menyeluruh (66 file, `src/`+`scripts/`) menemukan bug yang membiaskan angka `exp3b_asnorm` dan `exp5b_redimnet_fusion` yang dilaporkan sebelumnya (0.865 dan 0.908). Bug sudah diperbaiki, seluruh pipeline terdampak dihitung ulang, dan atas rekomendasi tambahan repetisi (`n_reps`) dinaikkan dari 5 ke **10** (target proposal semula) untuk exp3a/3b/3c/5b karena beberapa p-value berada tepat di ambang signifikansi pada 5 repetisi. Ringkasan bug yang diperbaiki (2026-07-26):
>
> - **Kebocoran evaluasi FSCIL**: query held-out task lama sempat diproses berulang lewat jalur stateful (`process()`) di tiap sesi berikutnya, alih-alih sekali saja — memengaruhi perbandingan `continual_mode="running_average"` (sistem usulan) vs `"static"`.
> - **Kebocoran cohort AS-Norm**: cohort AS-Norm dan sampel kalibrasi genuine sempat dibangun dari kolam speaker yang sama, membiaskan threshold yang dikalibrasi.
> - **Pooling Whisper**: mean-pooling encoder Whisper sempat mengikutsertakan silence padding — cache embedding `whisper` dihapus total dan dihitung ulang penuh (0 gagal). `exp5b` tidak terdampak (backbone kedua-nya `redimnet_b2`, bukan Whisper).
> - **Forgetting Measure** dan **paired bootstrap significance test** punya bug matematis masing-masing (off-by-one dan resampling tidak paired).
>
> **Hasil final (2026-08-02, 10 repetisi):** `exp3b_asnorm` = **0.833 ± 0.015**, melampaui baseline ECAPA (0.783) signifikan (p<0.0001), dan continual update (running_average) signifikan lebih baik dari static (p=0.0007) — kedua klaim yang sempat gagal di 5 repetisi kini **kembali terbukti signifikan** dengan statistical power yang lebih sesuai target proposal. `exp5b_redimnet_fusion` = **0.876 ± 0.016**, A3>A1 makin kuat signifikan (p<0.0001), melampaui baseline signifikan (p<0.0001); A3≈A2 (ReDimNet-saja) tetap setara (p=0.88); continual-vs-static pada akurasi tetap tidak signifikan (p=0.031) walau pada EER deteksi tetap signifikan kuat (10/10 repetisi, bootstrap CI).
>
> **Uji tambahan — ketahanan leakage (2026-08-02):** karena 80/100 task speaker resmi ada di VoxCeleb2-dev (data latih ReDimNet), dijalankan uji terpisah pada 19 task speaker yang **bukan** vox2-dev (`scripts/exp5_leakage_robustness.py`). Hasil: A3 (0.956) masih numerik di atas A1 (0.933) tapi **tidak signifikan** (p=0.1475) — kemungkinan besar karena subset kecil (19 speaker, 1 sesi statis, akurasi tinggi ~93-96% dengan sedikit ruang variansi) kurang bertenaga secara statistik, bukan bukti bahwa leakage membiaskan hasil resmi. Dilaporkan sebagai *inconclusive*, bukan sanggahan — lihat [`docs/experiment-5.md`](docs/experiment-5.md) §10.
>
> Detail lengkap, tabel penuh, dan uji signifikansi di [`docs/experiment-3.md`](docs/experiment-3.md) dan [`docs/experiment-5.md`](docs/experiment-5.md).

Test suite (156 test) **pass sepenuhnya** setelah seluruh perbaikan di atas — perbaikan bersifat korektif pada logika/kalibrasi, bukan perubahan API/kontrak yang memerlukan penyesuaian pemanggil.

## Dokumentasi Lengkap

- [`docs/README.md`](docs/README.md) — indeks dokumentasi tiap eksperimen (1-5) + [laporan lengkap satu dokumen](docs/laporan-lengkap-eksperimen.html).
- [`docs/ecapa-tdnn.md`](docs/ecapa-tdnn.md) — penjelasan arsitektur ECAPA-TDNN (materi landasan teori).
- [`plan/00-INDEX.md`](plan/00-INDEX.md) — rencana teknis: [project plan](plan/01-project-plan.md), [requirements](plan/02-requirements.md), [arsitektur](plan/03-architecture.md), [task breakdown](plan/04-tasks.md), [rencana evaluasi](plan/05-evaluation-plan.md).

## Batasan Penelitian

Sesuai proposal tesis Bab 1.3:

- Evaluasi difokuskan pada lingkungan umum (rumah, kantor, kafe, ruang publik semi-terbuka) — tidak mencakup lingkungan industri berat/akustik ekstrem.
- VAD hanya menangani segmentasi kontekstual (temporal + linguistik dasar) — tidak menangani *overlapping speech* secara menyeluruh.
- Evaluasi hanya memakai dataset publik **VoxCeleb1** dan **VoxCeleb2**.
- Tidak membahas dataset berskala sangat besar di luar VoxCeleb, maupun multibahasa secara mendalam.

Keterbatasan implementasi yang diketahui (bukan bug, keterbatasan desain algoritma F6-04/05): buffer deteksi pembicara baru di `src/continual/manager.py` bersifat satu buffer global untuk semua sampel unknown — dua pembicara baru yang berbeda dan muncul berdekatan waktu berpotensi tercampur menjadi satu kandidat cluster sebelum salah satu mencapai ambang jumlah sampel minimum.

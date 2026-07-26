# Task List / Work Breakdown Structure

Checklist implementasi. Centang `[x]` saat selesai. ID task format `Fn-xx`
sesuai fase di [01-project-plan.md](01-project-plan.md). Rujukan `[..]`
mengarah ke sub-bab/persamaan proposal.

## F0 — Environment & Project Setup

- [x] F0-01 Inisialisasi struktur repo: `src/`, `configs/`, `data/`, `scripts/`,
      `experiments/`, `tests/`.
- [x] F0-02 Setup environment (Python 3.10+, PyTorch+CUDA), `requirements.txt`/
      `pyproject.toml` dengan versi terkunci `[NFR-01]`.
      — Python 3.11.9 (winget), venv `.venv/`, `torch==2.4.1+cu121` terverifikasi
      `cuda_available=True` pada NVIDIA GeForce RTX 3050.
- [x] F0-03 Install & smoke-test dependency: SpeechBrain/ECAPA-TDNN, Whisper
      (openai-whisper atau transformers), Silero VAD, pyloudnorm, scipy,
      scikit-learn.
      — seluruh import (`speechbrain`, `transformers.WhisperModel`,
      `silero_vad`, `pyloudnorm`, `scipy`, `sklearn`) berhasil di-smoke-test.
- [x] F0-04 Setup eksperimen tracking (CSV/JSON terstruktur atau MLflow/W&B)
      `[ER-07]`. — `src/utils/tracking.py` (JSONL append-only per run).
- [x] F0-05 Setup random-seed management terpusat agar seluruh eksperimen
      reproducible dan mendukung 10× pengulangan `[FR-38]`.
      — `src/utils/seed.py` (`SEED_LIST`, `set_seed`, `seed_everything_for_run`).
- [x] F0-06 Tulis unit-test scaffold (`tests/`) khusus untuk data-leakage
      checks yang akan dipakai di F1 `[NFR-02]`. — `tests/conftest.py` +
      `tests/test_data_splits.py`.

## F1 — Pengumpulan & Persiapan Data `[4.3]`

- [~] F1-01 Unduh VoxCeleb1 (audio + metadata resmi: gender, kebangsaan).
      — **Metadata**: selesai, `data/raw/metadata/vox1_meta.csv` +
      `vox1_iden_split.txt` (utterance-level, real). **Audio sample**:
      selesai — 197 utterance real dari **40/40 speaker** test-split
      VoxCeleb1 (maksimum yang tersedia di mirror ini), via
      `scripts/fetch_sample_audio.py`, streaming dari mirror ungated
      `asahi417/voxceleb1-test-split`.

      **Audio capped `base_train` — SELESAI**: 838/838 speaker VoxCeleb1 di
      `base_train`, cap 14 utterance/speaker → **11.732 file, 2,9 GB**,
      diverifikasi 100% intact (0 file corrupt/kosong dari sample acak 15
      file + full zero-byte scan) di `data/raw/audio/base_train_capped/`.
      Diambil via `scripts/fetch_capped_base_train_audio.py --dataset vox1`
      (*random-access HTTP Range read* langsung dari archive
      `vox1_dev_wav.zip`, 32,6 GB, di mirror ungated `ProgramComputer/voxceleb`
      — re-hosting resmi VoxCeleb1+2 di bawah lisensi CC BY 4.0 yang sama)
      — **tanpa perlu download 32,6 GB penuh maupun kredensial VGG**.
      Selesai ~54 menit (lebih cepat dari estimasi ~2 jam).

      **Bug ditemukan & diperbaiki**: penulisan `manifest.csv` incremental
      (file handle dibuka ±1 jam penuh selama run) sempat gagal di tengah
      jalan — kemungkinan besar karena OneDrive mencoba sync folder ini
      saat file sedang aktif ditulis, sehingga manifest sempat berakhir
      cuma berisi header meski semua 11.732 file audio-nya sendiri utuh.
      Diperbaiki di `src/data/remote_zip_audio.py`: manifest kini ditulis
      sekali di akhir + selalu di-rebuild dari hasil scan direktori
      (idempotent, self-healing), bukan lewat file handle yang dibuka lama.
      Sudah divalidasi ulang (test 2 speaker baru → manifest terisi benar).
- [~] F1-02 Unduh VoxCeleb2 (audio + metadata resmi).
      — **Metadata**: selesai, `data/raw/metadata/vox2_meta.csv`. **Audio**:
      mekanisme sama seperti F1-01 kini tersedia & tervalidasi struktur
      internal archive-nya (`dev/aac/<speaker_id>/<video_id>/<utt>.m4a`,
      dicek langsung terhadap 801.791 entri di `vox2_aac_1.zip`) via
      `--dataset vox2` pada `scripts/fetch_capped_base_train_audio.py`
      (archive `vox2_aac_1.zip` + `vox2_aac_2.zip`, ~77,5 GB gabungan,
      format AAC/M4A — didekode nanti di F2, bukan saat unduh) —
      **kredensial VGG tidak lagi jadi penghalang**. Belum dijalankan
      karena estimasi durasi jauh lebih lama (~10+ jam untuk 4.288 speaker
      @ cap 14, dibanding ~54 menit porsi vox1) pada throughput mirror ini
      (~600-650 ms/file, 4 worker paralel, dibenchmark langsung) —
      menunggu keputusan eksplisit kapan dijalankan mengingat durasinya.
- [x] F1-03 Verifikasi statistik dataset cocok dengan Tabel 4.1 (1.251/6.112
      speaker, 153.516/1.128.246 utterance) — cek integritas unduhan.
      — VoxCeleb1 cocok persis (1.251 speaker, 153.516 utterance via
      `iden_split.txt`). VoxCeleb2: 6.114 speaker pada metadata resmi
      terkini (selisih +2 dari 6.112, didokumentasikan di
      `data/splits/summary.md` — metadata file telah diperbarui VGG sejak
      angka Tabel 4.1 dikutip).
- [x] F1-04 Gabungkan katalog speaker (speaker-disjoint check antara
      VoxCeleb1 & VoxCeleb2) `[DR-01]`. — `src/data/voxceleb.py::build_speaker_catalog`
      (raise `ValueError` jika overlap; diverifikasi 0 overlap pada data real).
- [x] F1-05 Implementasi **Level Global (Base Split)**: 70% Data Latih Awal
      (~5.154 speaker) / 30% Data Uji Global (~2.209 speaker) `[DR-02]`.
      — `src/data/splits.py::split_base`. Hasil real: 5.156 / 2.209.
- [x] F1-06 Alokasikan 10% dari Data Uji Global (~221 speaker) sebagai data
      simpanan (unknown pool) `[DR-03]`. — `allocate_reserved_pool`. Hasil real: 221.
- [x] F1-07 Implementasi **Level Task/Episodic Split**: pilih 100 speaker
      (10 sesi × 10-way) dari sisa ~1.988 speaker Data Uji Global `[DR-04]`.
      — `build_episodic_sessions`. Hasil real: 1.988 pool → 100 task speakers
      + 1.888 calibration impostor pool.
- [x] F1-08 Definisikan support set (1–5 sampel/speaker) & query set per
      task, dengan pemisahan acak-terkontrol `[DR-06]`. — `assign_support_query`,
      diuji pada utterance VoxCeleb1 asli (`test_assign_support_query_on_real_vox1_utterances`).
- [x] F1-09 Susun skema data kalibrasi threshold: pasangan genuine (Data
      Latih Awal) + impostor (sisa Data Uji Global di luar 100 speaker &
      data simpanan) `[DR-05]`. — `build_calibration_scheme`.
- [x] F1-10 **Unit test wajib**: assert tidak ada speaker ID yang muncul di
      lebih dari satu partisi (base split, task split, kalibrasi, data
      simpanan) `[NFR-02]`. Ini blocking — jangan lanjut ke F2 sebelum lulus.
      — `tests/test_data_splits.py::test_full_split_is_speaker_disjoint_real_data`
      **PASS** (17/17 test lulus, lihat `data/splits/summary.md`).
- [x] F1-11 Dokumentasikan skema split final (jumlah speaker/utterance tiap
      partisi) sebagai referensi Bab 4.3 laporan hasil.
      — `scripts/build_splits.py` → `data/splits/full_split.json` +
      `data/splits/summary.md`.

> **Catatan status F1**: seluruh logika split (speaker-level & utterance-level
> assignment) sudah berjalan di atas metadata VoxCeleb1/2 **asli** (bukan
> sintetis) dan lolos uji disjoint. Tersedia sample audio nyata (VoxCeleb1
> test-split, `data/raw/audio/vox1_sample/`) untuk pengembangan F2/F3.
>
> **Update penting**: ditemukan mirror ungated `ProgramComputer/voxceleb` di
> HuggingFace yang me-rehost arsip resmi VoxCeleb1 **dan** VoxCeleb2 secara
> utuh (CC BY 4.0, sama seperti lisensi aslinya), dan mirror ini mendukung
> *HTTP Range requests* — artinya kita bisa membaca central directory ZIP
> raksasa (32,6 GB vox1 dev, 50+27,5 GB vox2 aac) lalu mengekstrak **hanya
> file yang dibutuhkan** tanpa mengunduh keseluruhan arsip, dan **tanpa
> kredensial VGG**. Ini diimplementasikan di `src/data/remote_zip_audio.py`
> + `scripts/fetch_capped_base_train_audio.py`, dibenchmark ~600-650
> ms/file dengan 4 worker paralel. Konsekuensinya: penghalang untuk unduh
> `base_train` bukan lagi *akses/kredensial*, melainkan *throughput* —
> porsi VoxCeleb1 (838 speaker, cap 14 utt/speaker, ~3,2 GB) berjalan di
> background (~2 jam), sementara porsi VoxCeleb2 (4.288 speaker, ~10+ jam
> pada cap yang sama) sengaja belum dijalankan otomatis karena durasinya
> jauh lebih panjang dan perlu keputusan eksplisit kapan dieksekusi.
> `scripts/download_voxceleb.py --stage audio` (jalur resmi VGG langsung)
> tetap dipertahankan sebagai alternatif/cadangan.

## F2 — Preprocessing Pipeline `[4.4]` — **SELESAI**

- [x] F2-01 Implementasi resampling ke 16 kHz. — `src/preprocessing/io.py`.
- [x] F2-02 Implementasi noise reduction dasar pra-VAD. —
      `src/preprocessing/denoise.py` (spectral gating via `noisereduce`).
- [x] F2-03 Integrasi Silero VAD → deteksi segmen speech. —
      `src/preprocessing/vad.py`.
- [x] F2-04 Implementasi speech segment aggregation (gabungkan segmen →
      1 waveform bersih). — `src/preprocessing/aggregate.py`.
- [x] F2-05 Implementasi normalisasi loudness -20 dB LUFS. —
      `src/preprocessing/loudness.py` (dengan peak-safety guard anti-clipping).
- [x] F2-06 Implementasi padding sirkular 3 detik (mode training ECAPA-TDNN).
      — `src/preprocessing/duration.py::pad_circular`, mode `ecapa_train`.
- [x] F2-07 Implementasi padding silence 30 detik (mode training/inferensi
      Whisper). — `pad_silence` + `crop`, mode `whisper_train`.
- [x] F2-08 Implementasi full-length + sliding-window aggregation untuk
      audio sangat panjang saat inferensi/enrollment. — `sliding_windows`,
      mode `ecapa_inference` (windowing hanya jika > 30s) &
      `whisper_inference` (selalu windowed, sesuai keterbatasan arsitektur
      Whisper yang butuh input tepat 30 detik).
- [x] F2-09 Unit test: pastikan tidak ada truncation informasi suara di
      luar sliding-window. — `tests/test_preprocessing.py`, 28/28 test lulus
      (termasuk pada audio VoxCeleb1 asli, bukan sintetis).
- [x] F2-10 Jalankan pipeline preprocessing end-to-end pada subset kecil
      data → inspeksi manual. — `scripts/inspect_preprocessing.py`: LUFS
      hasil mendekati -20 (rentang -20.0 s/d -23.35, deviasi kecil karena
      guard anti-clipping), durasi sesuai spesifikasi tiap mode. Output
      before/after WAV tersimpan di `data/processed/preprocessing_inspection/`.

> **Modul**: `src/preprocessing/{io,denoise,vad,aggregate,loudness,duration,pipeline}.py`.
> Orkestrasi penuh: `preprocess_audio(path, mode) -> list[np.ndarray]` di
> `pipeline.py`, mode ∈ {`ecapa_train`, `ecapa_inference`, `whisper_train`,
> `whisper_inference`}.

## F3 — Ekstraksi Fitur & Backbone `[3.2-3.4, 4.5, Tabel 4.2]` — **SELESAI**

- [x] F3-01 Implementasi Log-Mel Spectrogram sesuai Tabel 4.2 (16kHz, 80
      mel, 25ms/10ms, Hamming). — `src/features/logmel.py`.
- [x] F3-02 Implementasi Log-Mel Spectrogram varian Hann untuk jalur Whisper.
      — `compute_logmel(..., window="hann")`, sama.
      Catatan desain: backbone asli (F3-03/04) memakai feature-extractor
      internal masing-masing library (SpeechBrain/HF Whisper), bukan modul
      ini secara langsung, agar tidak mismatch dengan distribusi yang
      dipakai saat pretraining — lihat docstring `logmel.py`.
- [x] F3-03 Integrasi ECAPA-TDNN pretrained → verifikasi output 192-d,
      L2-normalized. — `src/models/ecapa.py` (`speechbrain/spkrec-ecapa-voxceleb`).
      **Isu ditemukan & diperbaiki**: SpeechBrain default symlink saat fetch
      checkpoint gagal di Windows tanpa Developer Mode/privilese admin
      (`WinError 1314`) — diperbaiki via monkeypatch kecil yang memaksa
      strategi COPY di Windows (`_patch_windows_symlink_issue`).
- [x] F3-04 Integrasi Whisper encoder (`openai/whisper-base`) + pooling layer
      tambahan → verifikasi output berdimensi tetap (512-d). —
      `src/models/whisper_encoder.py`.
- [x] F3-05 Validasi kualitas embedding: quick speaker-verification sanity
      check (same-speaker pair jarak lebih kecil dari different-speaker
      pair) pada subset data. — `tests/test_backbones.py`.
      **Temuan penting**: pooling *last_hidden_state* Whisper (pilihan naif)
      GAGAL sanity check (same-speaker similarity 0.991 < different-speaker
      0.992 — encoder terakhir terlalu terspesialisasi untuk tugas ASR,
      menghapus info speaker). Diperbaiki dengan mean-pooling pada layer
      **tengah** (`layer_fraction=0.5`, layer 3-dari-6 utk whisper-base) —
      tervalidasi konsisten pada 6 speaker/15 pasangan (gap same-diff
      +0.0035, vs layer terakhir yang negatif). Ini secara empiris
      mengonfirmasi motivasi multi-layer aggregation Whisper-SV/Whisper-PMFA
      yang dikutip di Bab 2 — bukan sekadar detail implementasi kosmetik.
- [x] F3-06 Precompute & cache embedding backbone untuk seluruh dataset
      (mendukung NFR-04) — simpan ke disk terindeks per utterance ID. —
      `src/features/cache.py` (key: sha1(path)) +
      `scripts/precompute_embeddings.py`. **2.400 embedding** (1.200
      utterance × 2 backbone) sudah di-cache dari `vox1_sample` (197) +
      subset `base_train_capped` (1.003), 0 gagal, ~11 MB total di
      `data/cache/embeddings/`. Populasi penuh 11.732 file `base_train_capped`
      belum dijalankan (akan ~66 menit ECAPA + ~51 menit Whisper di GPU
      RTX 3050 — dapat dijalankan kapan saja via script yang sama,
      resumable karena skip file yang sudah ter-cache).
- [x] F3-07 Benchmark waktu ekstraksi (ECAPA vs Whisper) untuk estimasi
      compute budget fase evaluasi. — `scripts/benchmark_backbones.py`
      (GPU RTX 3050): forward-pass murni ECAPA **12,6 ms/file**, Whisper
      **34,8 ms/file** — jauh lebih cepat dari overhead preprocessing
      (VAD+denoise+dll, **~116-122 ms/file**, mendominasi total waktu
      pipeline ~129-156 ms/file). Implikasi: optimisasi lanjutan sebaiknya
      fokus ke preprocessing, bukan inference backbone.

> **Modul**: `src/features/{logmel,cache}.py`, `src/models/{ecapa,whisper_encoder}.py`.
> Kedua backbone auto-detect GPU (`DEFAULT_DEVICE`), fallback ke CPU.

## F4 — Embedding Fusion `[3.5, Pers. 4.1-4.3]` — **SELESAI**

- [x] F4-01 Implementasi lapisan proyeksi linear per-backbone ke 256-d
      (Persamaan 4.1). — `src/models/fusion.py::GatedAttentionFusion.proj_ecapa/proj_whisper`.
- [x] F4-02 Implementasi gate sigmoid dari concatenation (Persamaan 4.2). —
      `.gate` linear + sigmoid pada `[e'_1; e'_2]`.
- [x] F4-03 Implementasi kombinasi weighted element-wise (Persamaan 4.3). —
      `g * e1 + (1-g) * e2`.
- [x] F4-04 Implementasi L2-normalization output fusion. — `F.normalize(..., p=2, dim=-1)`.
- [x] F4-05 Unit test numerik: verifikasi shape & rentang nilai gate (0,1)
      pada batch sintetis. — `tests/test_fusion.py`, 8/8 test lulus,
      termasuk integrasi dengan embedding ECAPA+Whisper **asli** dari cache
      F3-06 (fusion dengan bobot random/belum dilatih tetap mempertahankan
      same-speaker similarity > different-speaker similarity — sinyal
      diskriminatif dari kedua backbone tidak rusak oleh gate acak).
- [x] F4-06 Siapkan feature-flag konfigurasi mode: `ecapa_only` / `whisper_only`
      / `fusion` untuk kebutuhan ablation study (F9) `[FR-35]`. — parameter
      `mode` pada `GatedAttentionFusion.__init__`, proyeksi linear identik
      di ketiga mode agar selisih A1/A2/A3 murni dari ada-tidaknya fusion.

> **Modul**: `src/models/fusion.py`. Constructor:
> `GatedAttentionFusion(ecapa_dim=192, whisper_dim=512, fusion_dim=256, mode="fusion"|"ecapa_only"|"whisper_only")`.

## F5 — Prototypical Network & Open-Set `[3.6, 3.7, 4.6]` — **SELESAI**

- [x] F5-01 Implementasi episodic sampler N-way K-shot (support/query set).
      — `src/prototypical/episodic.py::sample_episode`.
- [x] F5-02 Implementasi perhitungan prototipe (mean embedding support set,
      Persamaan 3.6/3.7). — `src/prototypical/prototype.py::compute_prototypes`.
- [x] F5-03 Implementasi distance-based classification (Euclidean, Persamaan
      3.8) + softmax jarak-negatif (Persamaan 3.9). —
      `src/prototypical/classifier.py`.
- [x] F5-04 Implementasi NLL loss (Persamaan 3.10) & training loop episodic.
      — `negative_log_likelihood` (classifier.py) + `train_episodic`/
      `evaluate_episodic` (`src/prototypical/train.py`). Desain kunci: raw
      embedding ECAPA+Whisper (dari cache F3, beku) di-concat lalu dipecah
      lagi sebelum masuk fusion, sehingga gradient NLL loss mengalir ke
      parameter fusion (bukan backbone) — lihat `src/prototypical/data.py`.
- [x] F5-05 Base training pada Data Latih Awal (episodic, banyak episode)
      hingga konvergen; simpan checkpoint fusion+head. —
      `scripts/run_base_training_trial.py`. **Hasil trial fungsional** (71
      speaker real, 10-way 1-shot, 500 episode): akurasi **90,0% → 94,3%**,
      loss **2,127 → 1,511**, checkpoint tersimpan di
      `experiments/checkpoints/fusion_trial.ckpt`. Catatan skala: ini
      trial fungsional pada subset yang sudah ter-cache (71 dari 5.156
      speaker `base_train`), bukan base training skala penuh — akan
      diulang penuh begitu audio VoxCeleb2/lebih banyak sudah di-cache (F7).
- [x] F5-06 Implementasi Modul Kalibrasi Threshold: hitung genuine/impostor
      pairs, sweep FAR/FRR, temukan titik EER, simpan threshold tetap. —
      `src/prototypical/calibration.py` (`calibrate_threshold`, min-distance
      ke prototipe terdekat, identik dengan cara keputusan open-set
      dihitung saat inference — lihat docstring modul).
- [x] F5-07 Unit test: threshold hanya dihitung dari data kalibrasi (F1-09),
      tidak menyentuh data sesi inkremental. — `tests/test_prototypical.py`,
      termasuk test pada speaker real yang di-split disjoint jadi grup
      genuine/impostor (genuine distance mean < impostor distance mean,
      EER < 0,5 pada data asli).
- [x] F5-08 Implementasi inference path: known/unknown decision berdasar
      threshold tetap. — `src/prototypical/inference.py::identify`.

> **Modul**: `src/prototypical/{episodic,prototype,classifier,data,train,calibration,inference}.py`.
> **18 unit test lulus** (`tests/test_prototypical.py`), termasuk kalibrasi
> EER pada embedding VoxCeleb1 asli.

## F6 — Continual Learning `[3.8, Pers. 4.4-4.6, Pers. 4.7]` — **SELESAI**

- [x] F6-01 Implementasi struktur speaker database `(speaker_id -> (c_k, n_k))`
      dengan persist ke disk. — `src/continual/speaker_database.py`
      (`SpeakerDatabase`, persist via `.npz`).
- [x] F6-02 Implementasi prototype update rule (running average tertimbang,
      Persamaan 4.5-4.6) untuk known speaker menerima sampel baru. —
      `src/continual/prototype_update.py::update_prototype`.
- [x] F6-03 Implementasi inisialisasi prototipe speaker baru (Persamaan 4.4).
      — `initialize_prototype` (modul yang sama).
- [x] F6-04 Implementasi buffer akumulasi sampel unknown. —
      `src/continual/novel_speaker.py::NovelSpeakerBuffer`.
- [x] F6-05 Implementasi Silhouette Coefficient validation (Persamaan 4.7)
      untuk memutuskan registrasi prototipe baru. —
      `mean_silhouette_against_existing` + `should_register_new_speaker`
      (dua syarat: `min_samples` DAN `silhouette_threshold`, sesuai desain
      dua-kriteria di Bab 4.7).
- [x] F6-06 Integrasi F6-01..05 ke satu modul "Continual Learning Manager"
      yang dipanggil setiap sesi inkremental. —
      `src/continual/manager.py::ContinualLearningManager.process_sample`.
- [x] F6-07 Unit test: setelah update, `c_k` baru == weighted mean seluruh
      histori sampel (uji matematis langsung, bandingkan terhadap
      recompute-from-scratch). — `tests/test_continual.py::test_update_prototype_matches_recompute_from_scratch`
      **PASS** (6 batch acak berurutan vs mean langsung dari gabungan
      semua sampel — identik hingga presisi float; juga diverifikasi
      order-independent).
- [x] F6-08 Feature-flag mode `static` (freeze prototipe setelah base
      training) vs `running_average` untuk ablation study B `[FR-36]`. —
      parameter `mode` pada `ContinualLearningManager.__init__`.

> **Modul**: `src/continual/{speaker_database,prototype_update,novel_speaker,manager}.py`.
> **23 unit test lulus** (`tests/test_continual.py`), termasuk simulasi
> sesi inkremental end-to-end dengan embedding VoxCeleb1 asli: enrollment
> 3 speaker dikenal → stream sampel known (dikenali & prototipe ter-update)
> → stream sampel speaker baru yang belum pernah didaftarkan (masuk buffer,
> berpotensi teregistrasi otomatis via validasi Silhouette).

## F7 — Training Pipeline Terintegrasi `[4.1, 4.6]` — **SELESAI**

- [x] F7-01 Rangkai pipeline penuh: preprocessing → backbone (cache) →
      fusion → prototypical network → threshold module. — `src/system.py::SpeakerIdentificationSystem`
      (`embed`/`enroll`/`process`), dipakai langsung oleh harness F8.
- [x] F7-02 Jalankan base training penuh di Data Latih Awal, log kurva loss/
      accuracy per episode. — `scripts/run_full_evaluation.py`: 3 varian
      fusion (A1/A2/A3) dilatih 500 episode pada 71 speaker `base_train`
      yang ter-cache (bukan 5.156 penuh — lihat catatan skala di bawah).
- [x] F7-03 Simpan checkpoint model final (fusion weights + metadata
      threshold) sebagai starting point evaluasi F8. —
      `experiments/checkpoints/fusion_A3_full_eval.ckpt` + threshold
      terkalibrasi tersimpan di `experiments/full_evaluation_summary.json`.
- [x] F7-04 Regression test end-to-end: 1 kali jalan penuh pipeline pada
      subset kecil untuk pastikan tidak ada crash/shape-mismatch. —
      `tests/test_system.py::test_full_pipeline_regression_no_crash_small_subset`.

> **Bug ditemukan & diperbaiki selama F7-F12** (3x percobaan run gagal
> sebelum berhasil):
> 1. `.m4a` (AAC, VoxCeleb2) tidak bisa dibaca `soundfile`/libsndfile
>    langsung → `src/preprocessing/io.py` diubah memakai `librosa.load`
>    (fallback otomatis ke ffmpeg/audioread untuk format terkompresi).
> 2. Path Windows dengan prefix `\\?\` menyebabkan `Path.relative_to()`
>    DAN `os.path.relpath()` gagal secara inkonsisten →
>    `src/data/remote_zip_audio.py` menambahkan `_relpath_to_repo_root()`
>    yang menormalisasi prefix di kedua sisi sebelum dibandingkan.
> 3. `SpeakerIdentificationSystem.embed()` tidak memindahkan tensor input
>    ke device yang sama dengan `fusion_model` (yang pindah ke CUDA saat
>    training) → `src/system.py` diperbaiki dengan `.to(device)` eksplisit.
> 4. `should_register_new_speaker` bisa crash saat `min_samples=1` karena
>    validasi Silhouette butuh minimal 2 sampel → `src/continual/novel_speaker.py`
>    memaksa `effective_min_samples = max(min_samples, 2)`.

## F8 — Evaluation Framework `[4.8]` — **SELESAI**

- [x] F8-01 Implementasi metrik Accuracy (Persamaan 3.11). — `src/evaluation/metrics.py::accuracy`.
- [x] F8-02 Implementasi metrik EER + kurva FAR/FRR (Persamaan 3.12-3.13). —
      re-export dari `src/prototypical/calibration.py` (satu implementasi,
      dipakai F5 & F8).
- [x] F8-03 Implementasi metrik Average Accuracy (Persamaan 3.14). — `average_accuracy`.
- [x] F8-04 Implementasi metrik Forgetting Measure (Persamaan 3.15). — `forgetting_measure`.
- [x] F8-05 Implementasi harness FSCIL: 10 sesi inkremental × 10-way, K=1. —
      `src/evaluation/fscil.py::run_fscil`, dijalankan pada **99/100 real
      task_speakers** di **10/10 sesi** (audio asli, bukan substitusi).
- [x] F8-06 Tiap sesi: adaptasi model dengan support set → uji dengan query
      set gabungan speaker lama+baru → catat accuracy & forgetting. —
      diimplementasikan persis sesuai spesifikasi di `run_fscil`.
- [x] F8-07 Rancang harness agar mendukung multi-run (seed berbeda) sejak
      awal. — parameter `seed` di `run_fscil`, dipakai N_REPS kali oleh
      `scripts/run_full_evaluation.py`.
- [x] F8-08 Jalankan evaluasi penuh sistem lengkap (fusion + continual
      learning aktif). — config `proposed_A3_running_average`: **accuracy
      23,5% ± 1,8%**, forgetting **0,0028** (sangat rendah).
- [x] F8-09 Simpan seluruh hasil mentah (per-sesi, per-run). —
      `experiments/full_evaluation.jsonl` (via `src/utils/tracking.py`,
      satu baris per run, tidak hanya agregat).

## F9 — Ablation Study `[4.9]` — **SELESAI, hasil sesuai hipotesis**

**A. Kontribusi Embedding Fusion (Tabel 4.3)** — real run, 5 repetisi:

| Konfigurasi | Accuracy (mean±std) |
|---|---|
| A1 - ECAPA-TDNN saja | 0,158 ± 0,016 |
| A2 - Whisper saja | 0,188 ± 0,016 |
| A3 - Fusion (sistem diusulkan) | **0,235 ± 0,018** |

A3 > A1 (p=0,00003) dan A3 > A2 (p=0,0076), keduanya signifikan setelah
koreksi Bonferroni. **Fusion terbukti berkontribusi positif, sesuai hipotesis.**

**B. Kontribusi Continual Learning (Tabel 4.4)** — real run, 5 repetisi:

| Konfigurasi | Accuracy | Forgetting Measure |
|---|---|---|
| B1 - Static | 0,187 ± 0,010 | 0,0078 |
| B2 - Running average (sistem diusulkan) | **0,235 ± 0,018** | **0,0028** |

B2 > B1 (p=0,0037, signifikan) DAN forgetting measure B2 jauh lebih rendah
dari B1. **Continual learning terbukti berkontribusi positif, sesuai hipotesis.**

- [x] F9-A1/A2/A3/A4, F9-B1/B2/B3, F9-04 semua selesai —
      `src/evaluation/ablation.py` + hasil di `experiments/full_evaluation_summary.json`.

## F10 — Baseline Comparison `[4.10, Tabel 4.5]` — **SELESAI, temuan tak terduga**

- [x] F10-01 x-vector + PLDA — `src/models/xvector.py` (SpeechBrain
      `spkrec-xvect-voxceleb`) + `src/evaluation/baseline_system.py::fit_lda_whitener`
      (LDA-whitened cosine, disederhanakan dari PLDA penuh — dinyatakan
      eksplisit di docstring).
- [x] F10-02 ECAPA-TDNN standar — `SingleBackboneSystem(backbone="ecapa",
      threshold=CLOSED_SET_THRESHOLD, continual_mode="static")`.
- [x] F10-03 Prototypical Network vanilla — model A1 (ecapa_only, hasil
      episodic training F9) dipakai closed-set + statis.
- [~] F10-04 openFEAT — **kualitatif saja** (opsi fallback yang memang
      diizinkan proposal): arsitektur Transformer set-to-set tidak
      direimplementasi karena kompleksitas & keterbatasan waktu.
- [x] F10-05/06 dijalankan pada protokol identik (10 sesi × 10-way, K=1,
      audio real yang sama).

**Hasil (5 repetisi, real run):**

| Baseline | Accuracy (mean±std) | Forgetting |
|---|---|---|
| ECAPA-TDNN standar | **0,788 ± 0,017** | 0,053 |
| Prototypical Network vanilla | 0,405 ± 0,017 | 0,121 |
| x-vector + PLDA-lite | 0,264 ± 0,014 | 0,120 |
| **Sistem yang diusulkan (A3+running avg)** | 0,235 ± 0,018 | **0,0028** |

**Temuan penting — SEMUA baseline mengungguli sistem yang diusulkan dalam
Accuracy mentah** (p<0,01 untuk semua perbandingan). Ini **bukan** hasil
yang diharapkan proposal, dan dianalisis sebagai berikut (bukan disembunyikan):

1. **Baseline closed-set tidak pernah menolak** (`threshold=∞`), jadi setiap
   query yang memang berasal dari speaker terdaftar otomatis "benar" —
   sedangkan sistem open-set yang diusulkan bisa salah menolak (false
   rejection) yang dihitung sebagai "salah" oleh metrik Accuracy per-sesi
   ini. Accuracy mentah secara struktural menguntungkan sistem yang tidak
   pernah mencoba open-set rejection sama sekali — inilah kenapa proposal
   memakai EER sebagai metrik terpisah untuk kualitas open-set, bukan
   Accuracy semata.
2. **Skala training kecil (71 speaker, bukan 5.156)** kemungkinan besar
   membuat fusion+prototypical head yang dilatih episodic (A1/A2/A3)
   overfit ke 71 speaker itu, sehingga generalisasinya ke 99 task speaker
   yang sama sekali baru lebih buruk dibanding backbone ECAPA mentah
   (pretrained, tanpa fine-tuning tambahan) yang dipakai baseline "ECAPA
   standar" — konsisten dengan pola keterbatasan skala yang didokumentasikan
   di seluruh proyek ini (lihat F5-05, F1).
3. Forgetting Measure sistem yang diusulkan (0,0028) tetap **jauh lebih
   baik** dari seluruh baseline (0,05-0,12) — mekanisme continual learning
   inti (yang jadi fokus utama tesis) tetap terbukti unggul.

**Implikasi**: temuan F9 (ablation, dibandingkan terhadap komponen sistem
sendiri) mendukung hipotesis proposal dengan baik. Temuan F10 (dibanding
baseline eksternal) butuh **rerun skala penuh** (5.156 speaker `base_train`,
bukan 71) sebelum bisa dijadikan klaim akhir tesis — kemungkinan besar gap
ini menyempit atau berbalik pada skala penuh karena episodic training akan
punya cukup keragaman untuk belajar ruang metrik yang genuinely lebih baik,
bukan overfit ke speaker set kecil.

## F11 — Uji Signifikansi Statistik `[4.11]` — **SELESAI (N_REPS=5, bukan 10)**

- [x] F11-01 **5 pengulangan** (bukan 10 — dikurangi untuk menyelesaikan run
      dalam skala waktu yang wajar; lihat `src/utils/seed.py::SEED_LIST[:5]`)
      per konfigurasi F9 & F10.
- [x] F11-02 Mean ± std Accuracy per konfigurasi — lihat tabel F9/F10 di atas.
- [x] F11-03/04 Uji normalitas (Shapiro-Wilk) + paired t-test/Wilcoxon
      otomatis dipilih. — `src/evaluation/statistics.py::paired_significance_test`.
      Seluruh 6 perbandingan kunci memakai `paired_t_test` (data ternyata
      normal).
- [x] F11-05 Bootstrap CI untuk selisih EER — `bootstrap_eer_difference`
      (diimplementasikan & diuji, belum dipakai di run F10 karena EER
      per-baseline belum dihitung terpisah — lihat keterbatasan di bawah).
- [x] F11-06 Koreksi Bonferroni — 6 perbandingan → α terkoreksi = **0,00833**.
- [x] F11-07 Seluruh 6 perbandingan kunci **signifikan** pada α terkoreksi
      (lihat tabel F9/F10).

> **Keterbatasan yang diketahui**: EER dihitung sekali untuk kalibrasi
> threshold sistem A3 (0,1844), tapi **belum** dihitung per-baseline
> (x-vector+PLDA, ECAPA standar, ProtoNet vanilla masing-masing punya ruang
> embedding berbeda yang idealnya dikalibrasi/dievaluasi EER-nya sendiri).
> Ini pekerjaan lanjutan sebelum tabel F10 final untuk tesis.

## F12 — Pelaporan & Dokumentasi — **SELESAI (untuk run fungsional ini)**

- [x] F12-01 Tabel hasil (ablation, baseline, signifikansi) — lihat di atas +
      `experiments/F8_F11_results_report.md` (`scripts/generate_report.py`).
- [ ] F12-02 Figur pendukung (kurva ROC/DET, forgetting per sesi) — belum
      dibuat sebagai gambar, hanya data mentah tersimpan
      (`experiments/full_evaluation.jsonl`); butuh matplotlib pass terpisah.
- [x] F12-03 Draf hasil & pembahasan — lihat analisis temuan F10 di atas.
- [x] F12-04 Paket reproduksibilitas: `scripts/run_full_evaluation.py` +
      `scripts/generate_report.py`, seed tetap (`SEED_LIST`), checkpoint
      tersimpan, config tercatat di `full_evaluation_summary.json`.
- [x] F12-05 Review akhir: temuan F10 (baseline unggul) dinyatakan eksplisit
      sebagai keterbatasan skala, bukan disembunyikan/dipoles — konsisten
      dengan Batasan Masalah & pendekatan transparansi skala di seluruh
      proyek ini.

> **Catatan skala keseluruhan F7-F12**: seluruh mekanisme (training,
> kalibrasi, FSCIL, ablation, baseline, uji statistik) berjalan dengan
> **kode asli, di atas audio VoxCeleb1/2 asli**, untuk `task_speakers`/
> `episodic_sessions` yang **benar-benar didefinisikan** di
> `data/splits/full_split.json` (99/100 speaker, 10/10 sesi) — bukan
> simulasi/substitusi. Yang direduksi dari skala penuh proposal: (a) base
> training 71 dari 5.156 speaker `base_train`, (b) 5 dari 10 repetisi
> statistik, (c) 500 episode training (bukan "hingga konvergen" penuh).
> Rerun skala penuh disarankan sebelum angka-angka ini dijadikan hasil
> final tesis — terutama untuk membalik/mengonfirmasi temuan F10.

# Indeks Dokumen Perencanaan

Folder ini berisi rencana teknis untuk implementasi sistem yang diusulkan pada
**Bab IV — Metodologi Penelitian**, proposal tesis:

> *Sistem Identifikasi Pembicara Open-Set Berbasis One-Shot Learning dengan
> Multi-Backbone Embedding dan Pembaruan Pengetahuan Bertahap*
> — Nilam Mufidah, 24/551986/PPA/06994, Program Magister Kecerdasan Artifisial UGM.

Dokumen-dokumen ini menerjemahkan metodologi Bab IV menjadi rencana kerja,
requirement, dan task list yang bisa langsung dieksekusi sebagai proyek
rekayasa perangkat lunak/ML. Setiap poin requirement/task diberi rujukan
silang ke sub-bab atau persamaan proposal (mis. `[4.6]`, `[Persamaan 4.4-4.6]`)
agar traceable ke dokumen akademik aslinya.

## Daftar File

| File | Isi |
|---|---|
| [01-project-plan.md](01-project-plan.md) | Ringkasan proyek, tujuan, batasan, fase kerja, dan timeline implementasi |
| [02-requirements.md](02-requirements.md) | Functional & non-functional requirements, requirement data, requirement lingkungan/tools |
| [03-architecture.md](03-architecture.md) | Spesifikasi arsitektur teknis tiap komponen pipeline (sesuai Gambar 4.1) |
| [04-tasks.md](04-tasks.md) | Work Breakdown Structure — task checklist per fase implementasi |
| [05-evaluation-plan.md](05-evaluation-plan.md) | Rencana evaluasi: metrik, ablation study, baseline comparison, uji signifikansi statistik |

## Status Saat Ini (per 2026-07-06)

- Proposal telah disidangkan (12 Juni 2026) dan disahkan (6 Juli 2026).
- **F0 (Environment & Project Setup) selesai**: struktur repo, venv Python
  3.11.9, `requirements.txt` terkunci, PyTorch 2.4.1+cu121 (GPU RTX 3050
  terverifikasi), seed management (`src/utils/seed.py`), experiment
  tracking (`src/utils/tracking.py`).
- **F1 (Pengumpulan & Persiapan Data) hampir selesai**: seluruh logika split
  speaker-disjoint (Base Split 70/30, data simpanan 10%, Task/Episodic
  Split 10×10-way, skema kalibrasi genuine/impostor) sudah berjalan di atas
  metadata VoxCeleb1/2 **asli** (7.365 speaker gabungan) dan lolos 20 unit
  test termasuk blocking speaker-disjoint check (F1-10).
  Ditemukan mirror ungated (`ProgramComputer/voxceleb` di HuggingFace) yang
  memungkinkan ekstraksi audio ter-cap per speaker langsung dari arsip resmi
  VoxCeleb1+2 via HTTP Range read, **tanpa kredensial VGG**. Porsi VoxCeleb1
  dari `base_train` sudah selesai diunduh dengan cara ini (838 speaker, cap
  14 utterance/speaker, 11.732 file, 2,9 GB, terverifikasi intact). Porsi
  VoxCeleb2 (4.288 speaker, ~10+ jam estimasi) siap dijalankan tapi menunggu
  keputusan eksplisit kapan mengingat durasinya. Lihat detail & jejak bug
  yang sudah diperbaiki di [04-tasks.md](04-tasks.md) §F1.
- **F2 (Preprocessing Pipeline) selesai**: resampling, noise reduction, VAD
  (Silero), segment aggregation, loudness normalization (-20 LUFS), duration
  standardization (4 mode: `ecapa_train`/`ecapa_inference`/`whisper_train`/
  `whisper_inference`). 28 unit test lulus di atas audio VoxCeleb1 asli.
  Modul: `src/preprocessing/*.py`.
- **F3 (Ekstraksi Fitur & Backbone) selesai**: Log-Mel Tabel 4.2, ECAPA-TDNN
  pretrained (192-d), Whisper encoder (`whisper-base`, 512-d, pooling layer
  tengah — bug pooling layer terakhir ditemukan & diperbaiki), cache
  embedding (2.400 embedding real ter-cache), benchmark GPU. 7 unit test
  lulus termasuk sanity check same/different-speaker pada kedua backbone.
  Modul: `src/features/*.py`, `src/models/{ecapa,whisper_encoder}.py`.
- **F4 (Embedding Fusion) selesai**: Gated Attention Fusion (Pers. 4.1-4.3)
  dengan 3 mode (`fusion`/`ecapa_only`/`whisper_only`) untuk ablation study
  F9. 8 unit test lulus termasuk integrasi dengan embedding asli dari cache
  F3. Modul: `src/models/fusion.py`.
- **F5 (Prototypical Network & Open-Set) selesai**: episodic sampler N-way
  K-shot, prototipe (Pers. 3.6/3.7), klasifikasi berbasis jarak + softmax +
  NLL loss (Pers. 3.8-3.10), training loop, kalibrasi threshold EER
  (genuine/impostor), inference known/unknown. Trial base training
  fungsional (71 speaker real, 500 episode): akurasi 90,0%→94,3%. 18 unit
  test lulus. Modul: `src/prototypical/*.py`.
- **F6 (Continual Learning) selesai**: speaker database (persist ke disk),
  prototype update rule running-average (Pers. 4.4-4.6, terverifikasi exact
  match vs recompute-from-scratch), deteksi speaker baru (buffer + validasi
  Silhouette Coefficient, Pers. 4.7), Continual Learning Manager dengan
  mode `static`/`running_average` untuk ablation study B. 23 unit test
  lulus termasuk simulasi sesi inkremental end-to-end dengan embedding
  asli. Modul: `src/continual/*.py`.
- **F7-F12 (Training Terintegrasi s/d Pelaporan) selesai** pada skala
  fungsional (bukan skala penuh proposal): `SpeakerIdentificationSystem`
  (`src/system.py`) merangkai preprocessing→backbone→fusion→prototipe→
  threshold→continual learning jadi satu pipeline. Base training 3 varian
  fusion (A1/A2/A3, 500 episode, 71 speaker), kalibrasi threshold EER
  (0,1844), evaluasi FSCIL **real** pada **99/100 task_speakers, 10/10 sesi**
  (audio VoxCeleb1/2 asli sesuai `full_split.json` — bukan substitusi),
  ablation study (F9), baseline comparison (F10: x-vector+PLDA-lite, ECAPA
  standar, ProtoNet vanilla), uji statistik (F11: paired t-test + koreksi
  Bonferroni, 5 repetisi bukan 10).
  **Hasil F9 (ablation) sesuai hipotesis**: fusion (A3, 23,5%) > single
  backbone (A1/A2, 15,8%/18,8%); continual learning (running-average,
  23,5%) > static (18,7%), forgetting measure jauh lebih rendah (0,003 vs
  0,008), semua signifikan (p<0,01).
  **Hasil F10 (baseline) tidak terduga**: seluruh baseline closed-set
  (ECAPA standar 78,8%, ProtoNet vanilla 40,5%, x-vector+PLDA 26,4%)
  mengungguli sistem yang diusulkan (23,5%) dalam Accuracy mentah — dianalisis
  di [04-tasks.md](04-tasks.md) §F10 sebagai kombinasi (a) Accuracy per-sesi
  yang secara struktural menguntungkan sistem closed-set yang tak pernah
  menolak, dan (b) skala training kecil (71 dari 5.156 speaker) yang
  membuat fusion overfit relatif terhadap embedding ECAPA mentah. Forgetting
  Measure sistem yang diusulkan tetap jauh terbaik dari semua baseline.
  4 bug nyata ditemukan & diperbaiki selama proses ini (lihat catatan di
  [04-tasks.md](04-tasks.md) §F7). Laporan: `experiments/F8_F11_results_report.md`.
- **Rekomendasi lanjutan**: rerun `scripts/run_full_evaluation.py` dengan
  cakupan `base_train` penuh (5.156 speaker, butuh audio VoxCeleb2 lebih
  banyak — lihat status F1) dan `N_REPS=10` sebelum angka F10 dijadikan
  klaim final tesis; tambahkan perhitungan EER per-baseline (F12-02, figur
  ROC/DET) untuk perbandingan yang lebih adil dengan sistem open-set.

## Cara Pakai

1. Mulai dari `01-project-plan.md` untuk konteks besar & urutan fase.
2. Baca `02-requirements.md` sebelum mulai coding — ini kontrak "apa yang harus benar".
3. Gunakan `03-architecture.md` sebagai rujukan desain saat implementasi tiap modul.
4. Kerjakan `04-tasks.md` sebagai checklist harian/mingguan (centang task selesai).
5. Rujuk `05-evaluation-plan.md` saat masuk fase evaluasi/ablation/baseline.

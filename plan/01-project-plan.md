# Project Plan

## 1. Ringkasan

**Judul**: Sistem Identifikasi Pembicara Open-Set Berbasis One-Shot Learning
dengan Multi-Backbone Embedding dan Pembaruan Pengetahuan Bertahap.

**Masalah yang diselesaikan** `[1.2]`:
- Sistem identifikasi pembicara konvensional butuh banyak data per individu dan
  berasumsi closed-set (semua pembicara sudah dikenal).
- Pembicara baru di dunia nyata sering hanya punya 1 sampel suara (one-shot).
- Sistem yang ada cenderung memaksa klasifikasi ke kelas terdaftar meski
  pembicara sebenarnya belum pernah didaftarkan (tidak ada mekanisme open-set
  yang baik).
- Menambah pembicara baru biasanya butuh retraining penuh, berisiko
  *catastrophic forgetting* terhadap pembicara lama.

**Solusi yang diusulkan** `[4.1]`: pipeline VAD → dual-backbone speaker
embedding (ECAPA-TDNN + Whisper encoder) → *gated attention fusion* →
*prototypical network* untuk klasifikasi berbasis jarak → *threshold*
open-set berbasis EER → pembaruan prototipe inkremental (continual learning)
tanpa retraining penuh.

## 2. Tujuan Penelitian `[1.4]`

1. Identifikasi pembicara akurat dari **satu sampel suara** (one-shot) pada
   kondisi audio yang mengandung gangguan alami (noise latar, gema, variasi
   intonasi/gaya bicara).
2. VAD adaptif-kontekstual untuk ekstraksi segmen ucapan yang akurat.
3. Continual learning: menambah pembicara baru tanpa menurunkan akurasi
   pembicara lama (minim *catastrophic forgetting*).
4. Open-set identification: membedakan pembicara terdaftar vs pembicara baru
   yang belum pernah didaftarkan, tanpa memaksakan klasifikasi ke kelas yang ada.

## 3. Batasan Masalah `[1.3]`

| # | Batasan |
|---|---|
| B1 | Pengujian difokuskan pada lingkungan umum (rumah, kantor, kafe, ruang publik semi-terbuka) dengan noise dalam rentang wajar — **tidak** mencakup lingkungan industri berat/akustik ekstrem. |
| B2 | VAD hanya menangani segmentasi kontekstual (temporal + linguistik dasar) — **tidak** menangani *overlapping speech* secara menyeluruh. |
| B3 | Evaluasi hanya memakai dataset publik **VoxCeleb1** dan **VoxCeleb2**. |
| B4 | Tidak membahas dataset berskala sangat besar di luar VoxCeleb, maupun multibahasa secara mendalam. |

Implikasi rekayasa: jangan alokasikan waktu untuk overlapping-speech
diarization, jangan menambah dataset lain di luar VoxCeleb tanpa alasan kuat,
dan jangan menguji pada audio noise ekstrem — itu di luar cakupan yang bisa
diklaim dalam laporan.

## 4. Fase Kerja (mapping dari Bab 4.2 "Tahapan Penelitian")

Tahap 1–2 (Studi Literatur, Penyusunan Proposal) **sudah selesai** — proposal
sudah disidangkan dan disahkan. Rencana ini berfokus pada tahap 3–6:

| Fase | Nama | Bab rujukan | Output |
|---|---|---|---|
| F0 | Environment & Project Setup | — | Repo terstruktur, environment reproducible |
| F1 | Pengumpulan & Persiapan Data | 4.3 | VoxCeleb1+2 terunduh, ter-split sesuai protokol |
| F2 | Preprocessing Pipeline | 4.4 | Modul VAD, silence removal, normalisasi, padding |
| F3 | Ekstraksi Fitur & Backbone | 3.2–3.4, 4.5 | Log-Mel pipeline, ECAPA-TDNN & Whisper embedding extractor |
| F4 | Embedding Fusion | 3.5, 4.5 | Gated Attention Fusion module (trainable) |
| F5 | Prototypical Network & Open-Set | 3.6–3.7, 4.6 | Episodic training, klasifikasi jarak, threshold EER |
| F6 | Continual Learning | 3.8, 4.7 | Prototype update, novel-speaker detection (buffer + silhouette) |
| F7 | Training Pipeline | 4.1, 4.6 | Base training + episodic fine-tuning + checkpoint |
| F8 | Evaluation Framework | 4.8 | Accuracy, EER, Average Accuracy, Forgetting Measure, harness FSCIL 10×10-way |
| F9 | Ablation Study | 4.9 | Tabel A1-A3 (fusion), B1-B2 (continual learning) |
| F10 | Baseline Comparison | 4.10 | x-vector+PLDA, ECAPA-TDNN standar, ProtoNet vanilla, openFEAT (simplified) |
| F11 | Uji Signifikansi Statistik | 4.11 | t-test/Wilcoxon, bootstrap EER, koreksi Bonferroni |
| F12 | Pelaporan & Dokumentasi | Bab V (Penyusunan Laporan) | Draft Bab IV hasil, tabel/figur akhir, paket reproduksibilitas |

Detail task per fase ada di [04-tasks.md](04-tasks.md).

## 5. Dependensi Antar Fase

```
F0 → F1 → F2 → F3 → F4 → F5 → F7 ─┬→ F8 → F9  ─┐
                         F6 ───────┘        F10 ├→ F11 → F12
                                            F9,F10 saling independen setelah F8
```

Catatan penting:
- **F5 (Prototypical Network) dan F6 (Continual Learning) saling bergantung**:
  mekanisme *update* prototipe (F6) memakai definisi prototipe yang sama
  dengan F5 (Persamaan 3.7 == basis Persamaan 4.4-4.6), jadi implementasikan
  F5 dahulu sebagai fungsi murni yang bisa dipakai ulang oleh F6.
- **F9 dan F10 baru bisa jalan setelah F8** (harness evaluasi) selesai, karena
  keduanya memakai protokol pengujian yang identik dengan evaluasi utama.
- **F11 berjalan paralel/menempel pada F9 & F10** — setiap konfigurasi di
  kedua studi itu perlu diulang 10× dengan seed berbeda, jadi rancang F8's
  harness agar mendukung multi-run sejak awal (jangan retrofit belakangan).

## 6. Timeline (selaras Tabel 5.1, disesuaikan status hari ini 2026-07-06)

| Fase | Bulan Bab V terkait | Estimasi durasi kerja |
|---|---|---|
| F0–F1 | Jan–Jun 2026 (Pengumpulan Data) | 1–2 minggu (unduh+verifikasi cepat karena publik) |
| F2–F4 | Mar–Agu 2026 (Analisis & Pemodelan) | 3–4 minggu |
| F5–F7 | Mar–Agu 2026 (Analisis & Pemodelan) | 4–6 minggu (termasuk base training, tuning) |
| F8 | Jul–Sep 2026 (Evaluasi Sistem) | 1–2 minggu |
| F9–F11 | Jul–Sep 2026 (Evaluasi Sistem) | 3–4 minggu (10 pengulangan × banyak konfigurasi = compute-heavy) |
| F12 | Jul–Des 2026 (Penyusunan Laporan) | berkelanjutan, mulai paralel sejak F8 |

Fase F9–F11 adalah titik risiko jadwal terbesar karena butuh 10 run
independen per konfigurasi (≥5 konfigurasi ablation + 4 baseline) — total
puluhan run penuh. Rencanakan compute budget & paralelisasi run sejak F8.

## 7. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Reimplementasi openFEAT (Transformer set-to-set) terlalu kompleks | Baseline tidak lengkap | Ikuti strategi bertingkat `[4.10]`: coba versi linear-adaptor sederhana → jika gagal, banding kualitatif saja → jika tetap gagal, nyatakan eksplisit di Batasan Masalah |
| Kalibrasi threshold EER bocor ke data pengujian | Hasil open-set bias/optimistis | Ikuti skema split kalibrasi terpisah tegas `[4.6 poin 4]`: genuine/impostor pairs HANYA dari Data Latih Awal + sisa Data Uji Global yang tidak dipakai sesi inkremental/data simpanan |
| Biaya komputasi 10× pengulangan tiap konfigurasi ablation+baseline | Timeline molor | Cache embedding hasil backbone (ECAPA/Whisper tidak berubah antar run jika backbone dibekukan), hanya re-run bagian stokastik (episodic sampling, fusion training jika dilatih ulang) |
| Whisper encoder dimensi besar (512-1280) & lambat pada CPU | Ekstraksi fitur lambat | Gunakan GPU, precompute & cache embedding sekali di awal, gunakan varian Whisper terkecil yang cukup representatif dulu untuk iterasi cepat |
| Data leakage antar Data Latih Awal / Data Uji Global / sesi episodic | Metrik tidak valid, ditolak reviewer | Implementasikan speaker-disjoint split sebagai unit test otomatis (assert tidak ada speaker ID yang muncul di >1 partisi) sejak F1 |

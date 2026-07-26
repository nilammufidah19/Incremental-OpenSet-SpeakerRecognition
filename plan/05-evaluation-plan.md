# Rencana Evaluasi

Rujukan utama: Bab 4.8 (Evaluasi/Validasi/Pengujian), 4.9 (Ablation Study),
4.10 (Baseline Comparison), 4.11 (Uji Signifikansi Statistik).

## 1. Metrik `[3.9]`

| Metrik | Formula | Fungsi |
|---|---|---|
| Accuracy | `(1/N) Σ 1[ŷ_i = y_i]` (Pers. 3.11) | Ketepatan klasifikasi umum |
| EER | titik FAR(θ)=FRR(θ) pada kurva ROC (Pers. 3.12-3.13) | Keseimbangan error open-set |
| Average Accuracy | `(1/T) Σ a_{T,i}` (Pers. 3.14) | Performa rata-rata lintas sesi continual learning |
| Forgetting Measure | `(1/(T-1)) Σ (a_{l,i} - a_{T,i})` (Pers. 3.15) | Tingkat catastrophic forgetting |

Semua metrik dihitung oleh modul terpusat (F8-01..04) agar konsisten dipakai
di evaluasi utama, ablation, dan baseline comparison — jangan duplikasi
implementasi metrik antar eksperimen.

## 2. Protokol Pengujian Utama `[4.8]`

1. **Speaker-disjoint split** wajib di seluruh level (base split, task
   split, kalibrasi threshold) — divalidasi via unit test F1-10.
2. **Level Global**: 70% Data Latih Awal (base training) / 30% Data Uji
   Global (evaluasi akhir, mencakup seluruh speaker semua sesi).
3. **Level Task/Episodic Split**: 10 sesi inkremental × 10-way, K=1
   (adaptasi protokol FSCIL Tao et al. 2020, awalnya 10×10-way 5-shot pada
   CUB200). Tiap sesi:
   - Support set (1-5 sampel/speaker) → adaptasi model (prototype init/update).
   - Query set (speaker lama + baru) → evaluasi.
   - Setelah tiap penambahan speaker baru: uji ulang gabungan speaker lama+baru
     → catat accuracy terkini & forgetting.
4. Threshold open-set dikalibrasi **sekali** sebelum sesi pertama (lihat
   03-architecture.md §7), dipakai tetap di seluruh 10 sesi.

## 3. Ablation Study `[4.9]`

### A. Kontribusi Embedding Fusion (Tabel 4.3)

| Konfigurasi | ECAPA-TDNN | Whisper | Gated Attention Fusion |
|---|---|---|---|
| A1 - ECAPA-TDNN saja | ✓ | ✗ | ✗ |
| A2 - Whisper saja | ✗ | ✓ | ✗ |
| A3 - Sistem lengkap | ✓ | ✓ | ✓ |

- Komponen lain (preprocessing, prototypical network, protokol 10×10-way)
  **dijaga identik** — selisih hasil murni dari ada/tidaknya fusion.
- Metrik pembanding: **Accuracy** dan **EER**.
- Preseden metodologis: Zhao et al. (2024, Whisper-PMFA) membandingkan
  langsung terhadap ECAPA-TDNN/ResNet34 tunggal pada protokol identik.

### B. Kontribusi Continual Learning (Tabel 4.4)

| Konfigurasi | Mekanisme Prototype | Diperbarui Tiap Sesi | Skema Evaluasi |
|---|---|---|---|
| B1 - Static | Dihitung sekali, dibekukan | ✗ | 10×10-way (identik) |
| B2 - Sistem lengkap | Running average | ✓ | 10×10-way (identik) |

- Metrik pembanding: **Average Accuracy** dan **Forgetting Measure**, diukur
  berkelanjutan sepanjang 10 sesi.
- Preseden metodologis: Wang et al. (2025, Continual Speech Learning)
  membandingkan gated-fusion layer terhadap fine-tuning konvensional dengan
  metrik Average Forgetting.

Kedua kelompok ablation memakai pembagian data **identik** dengan skema
evaluasi utama (§2).

## 4. Baseline Comparison `[4.10]`

"Capability ladder" — tiap baseline menambah satu kemampuan mendekati sistem
final:

| Baseline | Sumber | Kemampuan yang Diuji |
|---|---|---|
| x-vector + PLDA | Snyder et al. (2018) | Tidak ada (fondasi historis, closed-set, statis) |
| ECAPA-TDNN standar | Desplanques et al. (2020) | Backbone sama, tanpa kerangka one-shot/open-set/continual |
| Prototypical Network | Snell et al. (2017) | One-shot dasar, tanpa open-set & continual |
| openFEAT | Kishan et al. (2022) | One-shot + open-set, tanpa pembaruan prototipe berkelanjutan |
| Sistem yang diusulkan | — | One-shot + open-set + continual learning |

**Aturan wajib**: seluruh nilai baseline diperoleh dengan **menjalankan
ulang** (re-run) implementasi pada protokol identik (dataset, split,
10×10-way K=1, metrik Accuracy & EER) — **bukan** mengutip angka dari paper
asli (protokol paper asli berbeda-beda, mis. Prototypical Network awalnya
diuji pada data citra, openFEAT pada simulasi rumah tangga dengan split
berbeda).

**Strategi bertingkat untuk openFEAT** (arsitektur Transformer set-to-set
kompleks):
1. Coba reimplementasi disederhanakan (mis. modul adaptasi Transformer →
   lapisan adaptasi linear setara) — nyatakan penyederhanaan secara eksplisit.
2. Jika reimplementasi sederhana pun tidak feasible → perbandingan
   kualitatif saja (deskripsi desain & kemampuan), **tanpa** angka
   kuantitatif di tabel.
3. Jika bahkan itu tidak memungkinkan → nyatakan keterbatasan secara
   eksplisit di Batasan Masalah laporan akhir, dengan alasan metodologis.

Larangan: jangan pernah mencampur nilai performa dari protokol pengujian
berbeda ke satu tabel kuantitatif yang sama.

## 5. Uji Signifikansi Statistik `[4.11]`

**Kenapa perlu**: episodic testing memakai sampling acak per episode, jadi
selisih performa antar konfigurasi bisa jadi hanya kebetulan sampling.

**Prosedur**:
1. Setiap konfigurasi (ablation A1-A3, B1-B2; baseline x4 + sistem
   diusulkan) dijalankan **10 pengulangan independen** dengan random seed
   berbeda untuk episodic sampling.
2. Hasil dilaporkan **mean ± standard deviation**.
3. **Metrik Accuracy** (proporsi sederhana):
   - Uji normalitas: Shapiro-Wilk test.
   - Jika normal → paired t-test.
   - Jika tidak normal → Wilcoxon signed-rank test.
4. **Metrik EER** (gabungan FAR/FRR, tidak bisa didekati uji parametrik
   biasa):
   - Bootstrap resampling → interval kepercayaan 95% dari selisih EER antar
     konfigurasi (metodologi Bengio & Mariéthoz, 2004 — untuk metrik
     berbasis error otentikasi seperti EER/HTER).
   - Selisih signifikan jika nilai nol berada **di luar** interval
     kepercayaan 95%.
5. **Koreksi Bonferroni**: karena banyak perbandingan berpasangan sekaligus
   (A3 vs A1, A3 vs A2, B2 vs B1, sistem vs 4 baseline, dst.), α awal 0.05
   dibagi jumlah total perbandingan → ambang signifikansi terkoreksi dipakai
   untuk interpretasi akhir.

**Daftar perbandingan berpasangan yang wajib diuji** (hitung total untuk
Bonferroni di F11-06):
1. A3 vs A1 (fusion vs ECAPA-only)
2. A3 vs A2 (fusion vs Whisper-only)
3. B2 vs B1 (continual learning vs statis)
4. Sistem diusulkan vs x-vector+PLDA
5. Sistem diusulkan vs ECAPA-TDNN standar
6. Sistem diusulkan vs Prototypical Network vanilla
7. Sistem diusulkan vs openFEAT (jika kuantitatif tersedia, lihat §4)

→ minimal 6-7 perbandingan berpasangan → α terkoreksi ≈ 0,05/7 ≈ 0,0071
(hitung ulang persis sesuai jumlah final yang benar-benar dijalankan).

## 6. Format Pelaporan Hasil

- Tabel ablation (A & B) dan baseline: kolom **mean ± std**, kolom
  **signifikan? (ya/tidak, α terkoreksi)**.
- Simpan hasil mentah per-run (bukan hanya agregat) agar bisa diaudit ulang.
- Kurva ROC/DET kalibrasi threshold disimpan sebagai figur pendukung
  (bukti transparansi penentuan threshold, `[NFR-06]`).
- Kurva Average Accuracy & Forgetting Measure vs nomor sesi (1-10)
  ditampilkan sebagai grafik garis untuk B1 vs B2.

# Dokumentasi Eksperimen

Dokumentasi training & evaluasi sistem **Incremental Open-Set Speaker Recognition**. Setiap konfigurasi reportable ditandai sebagai satu *Experiment* dengan tag stabil, dikelola lewat skema feature-flag di [`src/experiments.py`](../src/experiments.py).

> 📄 **[Laporan Lengkap Eksperimen 1–5](laporan-lengkap-eksperimen.html)** — draft laporan utuh satu dokumen: data → split → preprocessing → ekstraksi fitur → evolusi arsitektur → protokol evaluasi → hasil & uji signifikansi seluruh eksperimen.
> 📝 **[Laporan Data Penelitian](laporan-data-penelitian.md)** — versi naratif bergaya bab proposal (bahasa formal, runtut, ber-nomor Bab 1–9): latar belakang rancangan data, prosedur akuisisi, rancangan pembagian beserta justifikasinya, penjaminan validitas, dan pembahasan keterbatasan. Siap dipakai sebagai bahan Bab 4.3 tesis.
> 📦 **[Data — sumber, akuisisi & skema split](data.md)** — dataset apa yang dipakai, diambil dari link mana (metadata VGG, mirror ungated HuggingFace), bagaimana split speaker-disjoint dilakukan, cakupan audio yang benar-benar diunduh, dan audit leakage VoxCeleb2-dev.
> 📚 **[ECAPA-TDNN — penjelasan arsitektur lengkap](ecapa-tdnn.md)** — materi landasan teori untuk pembelajar baru: TDNN, Res2Net, SE attention, MFA, ASP, AAM-softmax, dan peran ECAPA di tiap eksperimen.
> 🎓 **[Proses Training Model — Experiment 1–5](training-model.md)** — arsitektur training gradien yang dirancang (prototypical episodik) dan kenapa dibekukan; empat tahap *parameter fitting* yang menggantikannya (residual-init, cohort AS-Norm, kalibrasi threshold, prototype); flow diagram per tahap + alur satu run resmi; evolusi jalur pembelajaran exp0→exp5b.

## Daftar Eksperimen

| # | Tag | Ringkasan | Akurasi usulan | Dokumen |
|---|---|---|---|---|
| 0 | `baseline_v0` | Fusi random-init, 500 episode training | 0.235 | *(regresi acuan)* |
| 1 | `exp1_frozen_residual` | Fusi **residual-init + beku (0 episode)** | 0.731 | [MD](experiment-1.md) · [HTML](experiment-1.html) |
| 2a | `exp2a_scorefusion_L4` ✅ | Whisper L4 + score-level fusion — Whisper tak menambah nilai; threshold jadi tersangka utama | 0.733 (≈ exp1) | [MD](experiment-2.md) · [HTML](experiment-2.html) |
| 3a | `exp3a_lowfrr` ✅ | Titik operasi low-FRR saja — tidak cukup sendirian; mengungkap drift update (P5) & memperbaiki bug threshold ablasi (P4) | 0.732 (≈ exp1) | [MD](experiment-3.md) · [HTML](experiment-3.html) |
| 3b | `exp3b_asnorm` ✅ | **AS-Norm + low-FRR** — melampaui baseline ECAPA closed-set 0.788 (p=0.0015); deteksi unknown EER 0.111, AUROC 0.952; forgetting ~0 | 0.865 ± 0.008 | [MD](experiment-3.md) · [HTML](experiment-3.html) |
| 3c | `exp3c_dualmetric` ✅ | View pelaporan dua-metrik dari run exp3b (closed-set & deteksi berdampingan) | = 3b | [MD](experiment-3.md) · [HTML](experiment-3.html) |
| 4 | *(analisis, tanpa tag)* ✅ | **Gerbang keputusan fusi** — plafon oracle di ruang AS-Norm +0.022; score-fusion ternormalisasi < ECAPA-saja pada semua w → **fusi Whisper ditutup definitif** | — (tidak mengubah 3b) | [MD](experiment-4.md) · [HTML](experiment-4.html) |
| 5b | `exp5b_redimnet_fusion` ✅ **(aktif — hasil utama)** | **ECAPA + ReDimNet, fusi z-score dua-ruang (DualASNorm, w=0.3)** — **A3 > A1 signifikan untuk pertama kalinya (p=0.0041)**; deteksi EER 0.075, AUROC 0.972, TAR@1%FAR 0.748; WavLM gagal gerbang (0.29, negative result terdokumentasi); leakage vox2-dev diaudit & diungkap | **0.908 ± 0.019** | [MD](experiment-5.md) · [HTML](experiment-5.html) |

## Cara mengaktifkan sebuah eksperimen

Cukup ubah tag — tidak ada nilai yang di-hardcode di pipeline.

```powershell
# opsi A: edit ACTIVE_EXPERIMENT di src/experiments.py
# opsi B: override lewat environment variable (tanpa mengubah kode)
$env:ACTIVE_EXPERIMENT = "exp1_frozen_residual"
.venv\Scripts\python.exe scripts\run_full_evaluation.py
.venv\Scripts\python.exe scripts\generate_report.py
```

## Menambah eksperimen baru (Experiment 2, 3, …)

Tambahkan satu `ExperimentConfig` ke dict `EXPERIMENTS` di [`src/experiments.py`](../src/experiments.py), lalu arahkan `ACTIVE_EXPERIMENT` ke tag baru. Buat dokumen `docs/experiment-N.md` + `.html` mengikuti format Experiment 1.

## Struktur singkat pipeline

```
audio → preprocessing (16kHz, denoise, VAD, loudness, durasi)
      → backbone beku (ECAPA-TDNN 192-d + backbone kedua, cached)
        · exp0–3: Whisper encoder 512-d → Gated Attention Fusion (256-d)
        · exp5b (aktif): ReDimNet-b2 192-d → concat [ê_ecapa; ê_redimnet] (384-d)
      → normalisasi skor (feature flag):
        · exp3b: AS-Norm 1-ruang terhadap cohort base_train
        · exp5b: DualASNorm per-ruang, skor = 0.3·z_ecapa + 0.7·z_redimnet
      → keputusan jarak-prototype + threshold (EER, atau target-FRR sejak exp3a)
      → continual update / registrasi speaker baru
```

Detail lengkap ada di dokumen tiap eksperimen.

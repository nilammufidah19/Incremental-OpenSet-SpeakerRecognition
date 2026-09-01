# Memo Keputusan — Batas Operasional "Bebas-Pelatihan" dan Status Fusi LLR (F6-3)

**Kepada:** Pembimbing
**Dari:** Nilam Mufidah
**Tanggal:** 30 Agustus 2026
**Dibutuhkan:** satu keputusan biner mengenai di mana garis batasan "bebas-pelatihan" ditarik. Memo ini satu halaman; lampiran teknis di [`experiment-6-plan.md`](experiment-6-plan.md) §2 dan [`experiment-6.md`](experiment-6.md) §3 dan §7b.

---

## Latar

Tesis ini berpremis **bebas-pelatihan**: 0 episode pelatihan, backbone beku, tanpa *fine-tuning*. Premis tersebut selama ini belum dirumuskan secara operasional, sehingga satu komponen yang diusulkan — **fusi LLR** (regresi logistik 3–4 koefisien untuk menggabungkan dua skor) — tidak dapat diputuskan boleh-tidaknya.

Keputusan ini kini menjadi jalur kritis. Pengukuran plafon (Experiment 6 §7b) menunjukkan fusi ECAPA+ReDimNet menyisakan ruang **+0.041** di atas backbone tunggal terbaik, dan bobot tetap hanya memanen 41 % darinya. LLR adalah mekanisme standar untuk memanen sisanya.

## Fakta: komponen yang *sudah* di-fit pada `base_train` di semua run resmi

| Komponen | Jumlah parameter | Data fit | Dipakai sejak |
|---|---|---|---|
| Statistik cohort AS-Norm | 300 vektor rerata/σ | 300 utterance `base_train` | Experiment 3b |
| Ambang keputusan (`target_frr`) | 1 skalar | genuine `base_train` + impostor pool | Experiment 3a |
| Bobot fusi `w` | 1 skalar (dipilih sweep) | paruh validasi | Experiment 5b |
| Whitening/WCCN readout Whisper | statistik orde-2 | 1.190 utterance `base_train` | Experiment 6 |
| **Fusi LLR (diusulkan, F6-3)** | **3–4 skalar** | trial `base_train` | — |

Seluruhnya speaker-disjoint dari task evaluasi (audit kebocoran: Experiment 6 §10). Literatur speaker verification menyebut kategori ini *calibration and fusion*, bukan *training* (Brümmer dkk., *IEEE TASLP* 15(7), 2007; toolkit BOSARIS/FoCal).

## Pertanyaan yang dimintakan keputusan

Batasan bebas-pelatihan melarang **tingkat 0** (gradien pada penghasil embedding — backend AAM-softmax/LoRA; sudah dicoret, tidak diperdebatkan). Pertanyaannya satu:

> **Apakah garis ditarik antara tingkat 0–1, atau antara tingkat 1–2?**
>
> - Tingkat 1 = statistik & kalibrasi skor ber-parameter-sedikit yang di-fit pada `base_train` (seluruh tabel di atas).
> - Tingkat 2 = nol parameter di-fit sama sekali.

## Konsekuensi masing-masing jawaban

**Garis 0–1 (tingkat 1 diizinkan):** F6-3 LLR berjalan; sistem yang ada tetap sah tanpa perubahan. Ini konsisten dengan praktik literatur.

**Garis 1–2 (tingkat 1 dilarang):** F6-3 batal — **tetapi AS-Norm, ambang kalibrasi, bobot `w`, dan whitening ikut batal**, yang berarti hasil resmi Experiment 3b dan 5b (angka headline tesis 0.876/0.882) gugur dan harus dibangun ulang tanpa komponen-komponen itu.

## Yang tidak menunggu keputusan ini

Aturan fusi adaptif nol-parameter (tingkat 2, F6-3b) sudah dijalankan dan terbukti mengungguli bobot konstan (p = 0.0085); penyempurnaannya sedang berjalan. Keputusan memo ini hanya menentukan apakah LLR *juga* boleh dicoba.

---

**Mohon jawaban:** ☐ garis 0–1 (LLR boleh) ☐ garis 1–2 (LLR tidak boleh, dengan konsekuensi di atas)

*Tanggal & catatan keputusan akan dicatat di `experiment-6-plan.md` §2.*

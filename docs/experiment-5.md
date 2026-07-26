# Experiment 5 — Backbone Alternatif Pengganti Whisper (Fusi Multi-Backbone Heterogen)

**Status:** ✅ **SUDAH DIJALANKAN** (18–19 Juli 2026) — laporan lengkap: [`experiment-5.html`](experiment-5.html)
**Tag:** `exp5b_redimnet_fusion` ✅ *(kini tag aktif default)*
**Prasyarat:** ✅ [Experiment 4](experiment-4.md) — fusi Whisper ditutup (G4.2) → backbone kedua diganti.

> ## 🏆 Hasil utama (run resmi, 5 seed, 530 query unknown nyata)
> **Usulan (A3, fusi `0.3·z_ecapa + 0.7·z_redimnet`, running-average): Accuracy = 0.908 ± 0.019** — rekor baru (+0.043 di atas exp3b 0.865; +0.120 di atas baseline ECAPA 0.788, p=0.0005). **A3 > A1 signifikan untuk pertama kalinya (p=0.0041 < α=0.00833)** — klaim multi-backbone tesis akhirnya terbukti terukur. Deteksi unknown: **EER 0.075, AUROC 0.972, TAR@1%FAR 0.748**. Forgetting 0.0022.
>
> **Jalan eksekusi (gerbang demi gerbang):** audit leakage (80/100 task speaker di vox2-dev, diungkap) → kandidat #1 **WavLM GAGAL G5.1** (layer terbaik 0.2925 < 0.70; SSL beku butuh head terlatih — dilarang batasan bebas-pelatihan) → kandidat #2 **ReDimNet-b2 ft_lm LOLOS semua gerbang** (standalone 0.878; plafon +0.069; fusi snapshot 0.890 > kedua backbone) → sweep validasi kunci **w=0.3** (0.9183 vs ECAPA-only 0.8875, 3/3 seed) → run resmi.
>
> **Catatan jujur:** A3 vs A2 (ReDimNet-saja 0.902) p=0.209 — fusi *setara* backbone tunggal terkuat, tidak melampauinya; B2>B1 p=0.0128 (arah positif, tak lolos Bonferroni); aturan `mean` exp4c terkonfirmasi vs ECAPA-saja (0.082 < 0.111) tetapi skor produksi w=0.3 sudah menyerapnya (0.075).
> **Uji signifikansi (kepatuhan proposal Bab 4.11):** Accuracy — Shapiro-Wilk → paired t-test + Bonferroni ✅; **EER — bootstrap CI 95% (Bengio & Mariéthoz 2004) dijalankan pertama kali di exp5** ✅: usulan vs A1 ΔEER −0.036 [−0.047, −0.025] SIGNIFIKAN; **continual vs static ΔEER +0.101 [+0.088, +0.114] SIGNIFIKAN 5/5 rep** (melengkapi p=0.0128 Accuracy); vs 3 baseline semua signifikan; vs A2 setara. Deviasi tersisa vs proposal: 5 repetisi (bukan 10; deviasi functional-scale seragam sejak exp0) dan baseline ke-4 (reimplementasi SOTA) via jalur fallback kualitatif hal. 53 proposal.
> Artefak: `full_evaluation_summary_exp5b_redimnet_fusion.json` · `exp5_screening_redimnet_b2.json` · `exp5_validation_sweep.json` · `exp5_wavlm_layer_sweep.json` · `exp5_leakage_audit.json` · `exp5_detection_rules_exp5b_redimnet_fusion.json` · `exp5_bootstrap_eer_exp5b_redimnet_fusion.json`.

*Bagian di bawah ini adalah dokumen rencana asli (dipertahankan sebagai jejak metodologi); hasil aktual dan arsitektur final ada di [`experiment-5.html`](experiment-5.html).*
**Tag rencana:** `exp5a_<backbone>_standalone` *(eksplorasi)* · `exp5b_<backbone>_fusion` *(run resmi)*
**Batasan tetap:** strict 1-shot (K_SHOT = 1), bebas-pelatihan (0 episode, backbone beku, tanpa fine-tuning), protokol evaluasi = [Experiment 3](experiment-3.md) penuh.

---

## 0. Ringkasan eksekutif

Premis tesis ini adalah sistem **multi-backbone**. Experiment 1–4 membuktikan pasangan ECAPA+Whisper tidak memberi kontribusi fusi: A3 ≡ A1 (p = 1.0, exp1–3); plafon oracle di ruang mentah hanya +2.6–4.0% (exp2); dan [Experiment 4](experiment-4.md) menutupnya definitif di ruang skor produksi — **plafon oracle AS-Norm +0.0225 ± 0.0054 (whisper_l4) dan score-fusion ternormalisasi terbaik 0.8408 < ECAPA-saja 0.8417 pada semua bobot w** (kasus "Whisper benar & ECAPA salah" hanya 11/63 error, tanpa pola). Akar masalahnya bukan mekanisme fusi (embedding gated, score mentah, dan score ternormalisasi sudah dicoba), melainkan **pemilihan backbone kedua**: Whisper dioptimalkan untuk ASR/terjemahan/identifikasi bahasa — bukan untuk memisahkan identitas speaker — sebagaimana diakui literatur adaptasi Whisper-untuk-SV sendiri ([Whisper-SV, Li dkk. 2024](https://arxiv.org/abs/2407.10048)); pendekatan yang berhasil memakai Whisper untuk speaker ([WSI, 2025](https://arxiv.org/abs/2503.10446)) semuanya **melatih ulang/fine-tune backend** — melanggar batasan bebas-pelatihan tesis ini.

Experiment 5 mengganti backbone kedua dengan model yang memenuhi dua syarat komplementaritas:

1. **Kuat secara mandiri** pada speaker discrimination (Whisper-L4 hanya 0.216 open-set; pengganti harus mendekati kelas ECAPA).
2. **Heterogen terhadap ECAPA** — arsitektur (CNN-2D / SSL-transformer vs TDNN-1D) dan/atau paradigma pelatihan berbeda, sehingga pola error-nya berbeda. Preseden kuat: sistem juara VoxSRC menggabungkan ECAPA-TDNN dengan varian ResNet justru karena komplementaritas lintas-arsitektur ([IDLab VoxSRC-20](https://arxiv.org/abs/2010.11255); [ID R&D VoxSRC-22, juara Track 1–2 dengan fusi ResNet + model SSL](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2022/reports/ravana_idrnd.pdf)).

**Nilai acuan dari Experiment 4** (pembanding langsung setiap kandidat, protokol & script yang sama):

| Acuan (snapshot validasi, ruang AS-Norm, 3 seed) | Nilai |
|---|---|
| ECAPA-saja — akurasi identifikasi | **0.8417** |
| Whisper_l4-saja (backbone kedua saat ini) | 0.2425 |
| Δ plafon oracle ECAPA∪Whisper_l4 | +0.0225 ± 0.0054 |
| Score-fusion ternormalisasi terbaik (w=0.95) | 0.8408 (*di bawah* ECAPA-saja) |
| Deteksi: ECAPA-saja / aturan `mean` dua-ruang | EER 0.1654 / **0.1508** (AUROC 0.909 / 0.924; n unknown = 40) |

Kandidat pengganti dinyatakan menjanjikan bila ia **mengungguli Whisper di semua baris** — terutama standalone ≫ 0.24 dan Δ plafon ≫ +0.0225.

---

## 1. Kandidat backbone & dasar literatur

| # | Kandidat | Tipe / kontras vs ECAPA | Bukti performa | Sumber (pretrained) | Literatur |
|---|---|---|---|---|---|
| 1 | **ReDimNet** (b2/b6) | Topologi baru reshape 1D↔2D — kontras arsitektural kuat; 1–15 M param, efisien | SOTA speaker recognition Interspeech 2024 dengan parameter lebih sedikit | `torch.hub` [IDRnD/redimnet](https://github.com/IDRnD/redimnet) | [Yakovlev dkk., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/yakovlev24_interspeech.html) · [arXiv:2407.18223](https://arxiv.org/abs/2407.18223) |
| 2 | **WavLM** (base-plus, layer awal + agregasi multi-layer) | SSL 94k jam, transformer — paradigma pelatihan berbeda total; informasi speaker terkonsentrasi di layer awal (kebalikan Whisper) | SOTA SUPERB; backend WavLM mencapai EER < 1% Vox1-O (mis. WavLM-TDNN 0.52% dalam studi komparatif) | HuggingFace `microsoft/wavlm-base-plus` | [Chen dkk., IEEE JSTSP 2022](https://arxiv.org/abs/2110.13900) · [Chen dkk., ICASSP 2022 (SSL untuk ASV)](https://arxiv.org/abs/2110.05777) · [MMFA, *Electronics* 2025](https://doi.org/10.3390/electronics14193857) · [studi komparatif, *Applied Sciences* 2024](https://www.mdpi.com/2076-3417/14/4/1329) |
| 3 | **CAM++** | D-TDNN + context-aware masking — masih keluarga TDNN (kontras lebih lemah), tapi ringan & cepat | Mengungguli sistem arus utama di VoxCeleb & CN-Celeb dengan biaya komputasi lebih rendah | [3D-Speaker / modelscope](https://github.com/modelscope/3D-Speaker), WeSpeaker | [Wang dkk., Interspeech 2023, arXiv:2303.00332](https://arxiv.org/abs/2303.00332) |
| 4 | **ERes2NetV2** | Res2Net 2D dengan fusi fitur lokal–global — kontras arsitektural baik | Model terbaik di antara enam model 3D-Speaker-Toolkit | [3D-Speaker / modelscope](https://github.com/modelscope/3D-Speaker) | [ERes2NetV2, 2024](https://www.researchgate.net/publication/381158294_ERes2NetV2_Boosting_Short-Duration_Speaker_Verification_Performance_with_Computational_Efficiency) · [3D-Speaker-Toolkit, arXiv:2403.19971](https://arxiv.org/abs/2403.19971) |
| 5 | **ResNet34 (r-vector)** | CNN-2D klasik — kandidat "preseden VoxSRC" paling literal | EER ~1.0% Vox1-O; komponen fusi standar sistem juara | [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | [IDLab VoxSRC-20, arXiv:2010.11255](https://arxiv.org/abs/2010.11255) |

**Rekomendasi urutan uji:** **ReDimNet** (kontras arsitektur + SOTA + ringan + `torch.hub` tanpa dependency berat) → **WavLM-base-plus** (mempertahankan narasi asli tesis *"speech foundation model + speaker specialist"*, versi yang benar dari ide Whisper; **nol dependency baru** — `transformers` sudah terpasang, dan **bebas leakage VoxCeleb** — lihat verifikasi §2b yang bisa menaikkannya jadi #1) → CAM++/ERes2NetV2 sebagai cadangan. TitaNet (NeMo) sengaja tidak dimasukkan — dependency NeMo berat di Windows.

**Catatan narasi tesis:** kandidat #2 (WavLM) mempertahankan cerita "model fondasi ucapan umum digabung model spesialis speaker"; kandidat #1/#4/#5 menggeser cerita ke "fusi arsitektur heterogen" ala VoxSRC. Keduanya sah — pilih setelah melihat angka Fase 1–2.

---

## 2. Kriteria seleksi & audit anti-leakage (Fase 0)

**Syarat teknis kandidat:**
- Berjalan di PyTorch/Windows tanpa dependency eksotis; lisensi mengizinkan penggunaan riset.
- Menghasilkan embedding utterance-level dari audio 16 kHz secara **beku** (tanpa fine-tuning apa pun).
- Muat di GPU/CPU yang sama dengan pipeline sekarang (preseden: Whisper-L4 103 menit precompute — batas atas yang bisa diterima).

**Audit leakage (wajib dilaporkan di tesis) — temuan awal 18 Juli 2026:**
Pemeriksaan format ID pada `data/splits/full_split.json` menunjukkan mayoritas speaker evaluasi berasal dari **VoxCeleb2** (ID `id0xxxx` < 10001): **task_speakers 81/100**, `reserved_unknown_pool` 180/221, `calibration_impostor_pool` 1565/1888 (sisanya VoxCeleb1, `id1xxxx`). Karena hampir semua model speaker pretrained (WeSpeaker, 3D-Speaker, ReDimNet) dilatih pada **VoxCeleb2-dev**, kandidat-kandidat itu **kemungkinan besar pernah melihat sebagian besar speaker evaluasi saat pretraining**. Catatan konsistensi: ECAPA SpeechBrain yang dipakai sistem saat ini juga dilatih VoxCeleb1+2, jadi *perbandingan internal tetap adil* — tetapi irisan per-speaker harus diverifikasi eksplisit di Fase 0 (bandingkan daftar speaker terhadap metadata resmi VoxCeleb2 dev/test) dan dilaporkan di tesis. Kandidat **WavLM base-plus (pretraining Libri-Light/GigaSpeech/VoxPopuli — tanpa VoxCeleb) praktis bebas isu ini** — nilai plus besar untuk defensibility, dan bisa membalik urutan uji ReDimNet↔WavLM bila irisan ReDimNet terbukti besar.

---

## 2b. Verifikasi kesiapan — SUDAH DICEK (18 Juli 2026)

Hal-hal yang tadinya asumsi rencana dan kini sudah diverifikasi langsung di lingkungan proyek:

| Item | Status | Temuan |
|---|---|---|
| Infrastruktur screening (Fase 1–2) | ✅ siap | `scripts/exp4_complementarity_asnorm.py --second-backbone <nama> --out <file>` diimplementasikan & diuji (`--help` jalan, 7 unit test lolos); protokol identik dengan angka acuan exp4 di §1 |
| Precompute embedding kandidat | ✅ siap | `scripts/precompute_embeddings.py --manifest … --backbone <nama>` sudah generik; hanya butuh registrasi kandidat di `BACKBONE_MODES`/`BACKBONE_EXTRACTORS` (`src/features/cache.py`) + fungsi ekstraktor |
| Dependency WavLM | ✅ **nol dependency baru** | `transformers 4.45.2` sudah terpasang di venv (dipakai Whisper) → `WavLMModel` langsung bisa; `torchaudio 2.4.1+cu121`, `speechbrain 1.0.2` juga tersedia |
| Dependency ReDimNet | ⚠️ perlu internet saat pertama | dimuat via `torch.hub` (repo IDRnD) — tanpa pip install baru, tetapi perlu koneksi & pin revisi untuk reproducibility |
| Dependency CAM++ / ERes2NetV2 | ⚠️ dependency baru | butuh paket modelscope/wespeaker → tetap kandidat cadangan |
| Audit leakage awal | ⚠️ **risiko nyata** | 81/100 task speaker & 180/221 reserved pool ber-ID VoxCeleb2 → kandidat yang dilatih VoxCeleb2-dev kemungkinan pernah melihat mereka (§2); WavLM bebas isu ini |
| Cache `whisper_l4` paruh-validasi | ✅ terisi | 540 utterance dihitung saat exp4 → screening ulang variannya kini 0.2 menit |
| GPU | ✅ | CUDA tersedia (RTX 3050) — estimasi precompute kandidat ±1–2 jam realistis |

**Implikasi urutan uji:** ReDimNet tetap kandidat #1 secara performa/kontras arsitektur, tetapi jika audit per-speaker Fase 0 mengonfirmasi irisan besar dengan VoxCeleb2-dev, **WavLM naik menjadi #1** (nol dependency baru + bebas leakage + narasi "foundation model + specialist" asli tesis tetap hidup).

---

## 3. Rencana bertahap (fase + gerbang keputusan)

| Fase | Kegiatan | Keluaran / Gerbang | Estimasi |
|---|---|---|---|
| **0. Setup & audit** | **(0a)** Audit leakage per-speaker (§2/§2b) — unduh metadata resmi VoxCeleb2 dev/test, cocokkan dengan `full_split.json`; **(0b)** registrasi backbone terpilih (modul ekstraktor pola `whisper_encoder.py` + entri `BACKBONE_MODES`/`BACKBONE_EXTRACTORS` di `cache.py`); ekstraksi embedding subset kecil (±20 speaker) | Audit menentukan kandidat #1 (lihat urutan aksi di bawah); sanity: embedding deterministik, dimensi benar, cosine same-speaker > cross-speaker | ½ hari |
| **1. Standalone + screening** | Registrasi backbone di `BACKBONE_MODES`/`BACKBONE_EXTRACTORS` (`src/features/cache.py`) + modul ekstraktor; precompute cache (`scripts/precompute_embeddings.py --backbone <nama>`); lalu **langsung Fase 2** — script screening sudah melaporkan akurasi standalone (`acc_b`) sekaligus | **G5.1:** acc standalone (snapshot AS-Norm, dilaporkan script exp4) ≥ **0.70** — Whisper_l4 hanya 0.2425, ECAPA 0.8417. Gagal → kandidat berikutnya | ½–1 hari (precompute ±1–2 jam) |
| **2. Plafon komplementaritas** | `scripts/exp4_complementarity_asnorm.py --second-backbone <nama> --out experiments/exp5_screening_<nama>.json` — **flag sudah diimplementasikan & diuji** (18 Jul 2026); protokol & task identik screening exp4 sehingga angkanya langsung sebanding dengan tabel acuan §1 | **G5.2:** `delta_ceiling` AS-Norm ≥ **+0.03** (jelas di atas Whisper +0.0225) **dan** fusion sweep menunjukkan ∃w yang > ECAPA-saja (kegagalan justru di titik ini yang mengubur Whisper di G4.2). Gagal → kandidat berikutnya | ½ hari |
| **3. Sweep fusi (validasi FSCIL)** | Score-fusion ternormalisasi (formula exp4b), sweep `w` halus di sekitar w terbaik Fase 2; AS-Norm per-backbone (default c300/k200, re-sweep `top_k` bila perlu); threshold per-w (P4) — pada task FSCIL validasi lengkap (10×10-way, continual, pola sweep exp3), 3 seed | **G5.3:** val-acc fusi > val-acc ECAPA-only **FSCIL** (0.8867 — angka sweep exp3, *bukan* 0.8417 snapshot) secara konsisten antar seed → **kunci** hyperparameter | 1 hari |
| **4. Run resmi** | Tag `exp5b_<backbone>_fusion`, 5 seed, protokol exp3 penuh (10×10-way, 530 query unknown paruh-deteksi, Bonferroni α = 0.00833); **sekaligus uji aturan deteksi dua-ruang `mean` (temuan exp4c: EER validasi −0.015 pada n=40) terhadap 530 unknown** | Kriteria sukses §4 | ½ hari |
| **5. Dokumentasi** | `experiment-5` hasil + update `README.md` + HTML | Laporan lengkap, positif maupun negatif | ½ hari |

Total estimasi per kandidat: **3–4 hari kerja**; Fase 0–2 saja (gerbang murah) ±1,5 hari — kandidat yang gagal gerbang berhenti di situ.

### 3.1 Urutan aksi konkret — apa yang dikerjakan terlebih dahulu

1. **Audit leakage per-speaker (Fase 0a, ±30 menit, butuh internet)** — langkah PERTAMA karena hasilnya menentukan kandidat mana yang layak diinvestasikan precompute 1–2 jam: irisan besar `task_speakers`/`reserved_unknown_pool` dengan VoxCeleb2-dev → ReDimNet/CAM++ bermasalah untuk defensibility → **WavLM naik jadi #1**; irisan kecil → ReDimNet tetap #1. Angka irisan wajib dilaporkan di tesis apa pun hasilnya (termasuk untuk ECAPA SpeechBrain yang dipakai sekarang).
2. **Registrasi backbone terpilih (Fase 0b, ±1–2 jam kerja)** — modul ekstraktor + registrasi `cache.py` + sanity subset ±20 speaker.
3. **Precompute cache penuh (Fase 1, ±1–2 jam GPU, sekali saja)** — 3 manifest (`eval_capped`, `vox1_sample`, `base_train_capped`).
4. **Screening satu perintah (Fase 2, menit)** — script exp4 dengan `--second-backbone` → langsung memberi G5.1 (standalone), G5.2 (plafon + sweep fusi), dan deteksi dua-ruang, sebanding apel-ke-apel dengan tabel acuan Whisper (§1).

> **Jalur pintas rendah-risiko:** langkah 1 dan 2 boleh paralel dengan langsung memilih **WavLM** — aman dari kedua sisi (nol dependency baru, bebas leakage VoxCeleb); audit tetap dijalankan untuk pelaporan tesis dan untuk memutuskan apakah ReDimNet layak diuji sebagai kandidat kedua.

---

## 4. Kriteria keberhasilan run resmi (exp5b)

| Metrik | Pembanding | Target |
|---|---|---|
| **A3 (fusi) vs A1 (ECAPA saja)** | inti klaim multi-backbone | **A3 > A1, p < 0.00833** — untuk pertama kalinya fusi harus terukur, bukan ≡ |
| Open-set Acc | exp3b 0.865 ± 0.008 | > 0.865 (minimal tidak turun) |
| Closed-set Acc | 0.865 | > 0.865 |
| Forgetting | 0.0022 | tetap ~0 (< 0.01) |
| det-EER / AUROC / TAR@1%FAR | 0.111 / 0.952 / 0.594 | tidak memburuk; TAR@1%FAR adalah ruang perbaikan terbesar |
| A2 (backbone kedua saja) | Whisper 0.216 | ≥ 0.70 (bukti kualitas standalone) |
| Deteksi aturan `mean` dua-ruang (temuan exp4c) | EER validasi 0.151 vs 0.165 (n=40, belum konklusif) | konfirmasi/tolak pada 530 unknown paruh-deteksi; dipromosikan ke produksi hanya bila EER < baseline satu-ruang secara konsisten |

---

## 5. Perubahan implementasi yang diantisipasi

Semua di belakang feature flag, default = perilaku lama (disiplin exp3 §2):

| File | Perubahan |
|---|---|
| `src/features/cache.py` | registri varian backbone generik (perluasan pola `whisper_l4`): `redimnet_b2`, `wavlm_bp_l<k>`, dst. — namespace cache terpisah per varian |
| `src/features/` *(modul baru)* | ekstraktor embedding kandidat (loader `torch.hub`/HF, mean-pool bila perlu, L2-norm konsisten) |
| `scripts/precompute_embeddings.py` | ✅ **tidak perlu diubah** (diverifikasi 18 Jul 2026): flag `--backbone <nama>` sudah generik & memakai path absolut (pelajaran bug cache exp3 §2.3) — kandidat cukup diregistrasi di `cache.py` |
| `scripts/exp4_complementarity_asnorm.py` + `src/evaluation/complementarity.py` | ✅ **siap dipakai** (diverifikasi 18 Jul 2026): flag `--second-backbone`/`--out` untuk screening Fase 1–2 sudah diimplementasikan & diuji |
| `src/experiments.py` | field `second_backbone` (default `"whisper"` = perilaku lama); tag `exp5a_*`, `exp5b_*` |
| `src/system.py`, `src/prototypical/data.py` | parameter backbone kedua diteruskan (pola `whisper_backbone` exp2) |
| `src/continual/manager.py`, `src/prototypical/calibration.py` | reuse jalur skor dua-backbone ternormalisasi dari Experiment 4b |
| `tests/test_experiment5.py` | unit test: determinisme embedding, namespace cache, orientasi skor fusi, gerbang konfigurasi |

---

## 6. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| Leakage data latih backbone (VoxCeleb2-dev) | Audit Fase 0 + pelaporan eksplisit; preferensi WavLM (non-VoxCeleb) bila irisan besar |
| Biaya precompute per kandidat (~1–2 jam) | Gerbang bertingkat: subset dulu (Fase 0), cache penuh hanya setelah sanity lolos |
| Konflik dependency (torch/transformers versi) | Kandidat dibatasi `torch.hub`/HF/pip murni; NeMo dihindari; uji di venv terpisah dulu bila ragu |
| Semua kandidat gagal G5.2 (plafon kecil bahkan untuk backbone kuat) | Ini pun temuan valid: dengan AS-Norm, satu backbone kuat sudah menangkap hampir semua sinyal → tesis melaporkan studi ablasi multi-backbone dengan plafon terukur, dan klaim kontribusi berpindah sepenuhnya ke pipeline threshold/AS-Norm + continual update (sudah > baseline secara signifikan) |
| Dua backbone sama-sama benar tapi lewat jalur berbeda → fusi menggeser kalibrasi threshold | Kalibrasi per-konfigurasi (P4) + validasi terpisah sudah menjadi prosedur baku |

---

## 7. Daftar pustaka pendukung

1. Yakovlev, I., dkk. (2024). **Reshape Dimensions Network for Speaker Recognition** (ReDimNet). *Interspeech 2024*. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/yakovlev24_interspeech.html) · [arXiv:2407.18223](https://arxiv.org/abs/2407.18223) · [kode & bobot](https://github.com/IDRnD/redimnet)
2. Chen, S., dkk. (2022). **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing**. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505–1518. [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
3. Chen, Z., dkk. (2022). **Large-scale Self-Supervised Speech Representation Learning for Automatic Speaker Verification**. *ICASSP 2022*. [arXiv:2110.05777](https://arxiv.org/abs/2110.05777)
4. **MMFA: Masked Multi-Layer Feature Aggregation for Speaker Verification Using WavLM** (2025). *Electronics*, 14(19), 3857. [DOI:10.3390/electronics14193857](https://doi.org/10.3390/electronics14193857)
5. Wang, H., dkk. (2023). **CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking**. *Interspeech 2023*. [arXiv:2303.00332](https://arxiv.org/abs/2303.00332)
6. **ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency** (2024). [ResearchGate](https://www.researchgate.net/publication/381158294_ERes2NetV2_Boosting_Short-Duration_Speaker_Verification_Performance_with_Computational_Efficiency); toolkit: **3D-Speaker-Toolkit** (2024), [arXiv:2403.19971](https://arxiv.org/abs/2403.19971)
7. Thienpondt, J., Desplanques, B., Demuynck, K. (2021). **The IDLab VoxSRC-20 Submission: Large Margin Fine-Tuning and Quality-Aware Score Calibration** — fusi skor 6× ECAPA-TDNN + 4× ResNet34. *ICASSP 2021*. [arXiv:2010.11255](https://arxiv.org/abs/2010.11255)
8. ID R&D (2022). **System Description, VoxCeleb Speaker Recognition Challenge 2022** — juara Track 1 & 2 dengan fusi ResNet dalam + model SSL. [PDF laporan resmi VoxSRC](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2022/reports/ravana_idrnd.pdf)
9. Li, L., dkk. (2024). **Whisper-SV: Adapting Whisper for Low-data-resource Speaker Verification** — mengakui Whisper dioptimalkan untuk ASR/ST/LID, bukan SV. [arXiv:2407.10048](https://arxiv.org/abs/2407.10048)
10. **Whisper Speaker Identification (WSI): Leveraging Pre-Trained Multilingual Transformers for Robust Speaker Embeddings** (2025) — Whisper bisa dipakai untuk speaker *dengan fine-tuning* (di luar batasan tesis ini). [arXiv:2503.10446](https://arxiv.org/abs/2503.10446)
11. **Comparison of Modern Deep Learning Models for Speaker Verification** (2024). *Applied Sciences*, 14(4), 1329 — benchmark lintas model (ResNet34 EER 1.02%, WavLM-TDNN 0.52% di Vox1-O). [MDPI](https://www.mdpi.com/2076-3417/14/4/1329)
12. Desplanques, B., dkk. (2020). **ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification**. *Interspeech 2020*. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)

---

## 8. Reproduksi (rencana perintah)

```powershell
# Fase 0/1 — registrasi kandidat di src/features/cache.py, lalu precompute
# (flag --backbone & --manifest sudah tersedia; contoh: ReDimNet)
.venv\Scripts\python.exe scripts\precompute_embeddings.py `
    --manifest data/raw/audio/eval_capped/manifest.csv `
    --manifest data/raw/audio/vox1_sample/manifest.csv `
    --manifest data/raw/audio/base_train_capped/manifest.csv `
    --backbone redimnet_b2

# Fase 1+2 — standalone (acc_b) + plafon + sweep fusi + deteksi dua-ruang,
# satu perintah (script exp4, flag sudah diimplementasikan & diuji):
.venv\Scripts\python.exe scripts\exp4_complementarity_asnorm.py `
    --second-backbone redimnet_b2 --out experiments/exp5_screening_redimnet_b2.json

# Fase 3 — sweep fusi FSCIL di paruh-validasi (perluasan sweep exp3; dibuat saat G5.2 lolos)

# Fase 4 — run resmi (tag dibuat di src/experiments.py saat G5.3 lolos)
$env:ACTIVE_EXPERIMENT = "exp5b_redimnet_fusion"
.venv\Scripts\python.exe scripts\run_full_evaluation.py
```

---

*Rencana Experiment 5. Terkait: [experiment-2.md](experiment-2.md) · [experiment-3.md](experiment-3.md) · [experiment-4.md](experiment-4.md) · [README.md](README.md) · `src/experiments.py`.*

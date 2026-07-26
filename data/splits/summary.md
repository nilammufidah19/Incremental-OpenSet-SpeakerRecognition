# Ringkasan Split Data (F1-11)

Seed: `0`

| Partisi | Jumlah Speaker |
|---|---|
| Total gabungan (vox1+vox2) | 7365 |
| &nbsp;&nbsp;- VoxCeleb1 | 1251 |
| &nbsp;&nbsp;- VoxCeleb2 | 6114 |
| Data Latih Awal (base_train, 70%) | 5156 |
| Data Uji Global (30%) | 2209 |
| &nbsp;&nbsp;- Data simpanan (reserved, 10% of Data Uji Global) | 221 |
| &nbsp;&nbsp;- Episodic pool (90% of Data Uji Global) | 1988 |
| &nbsp;&nbsp;&nbsp;&nbsp;- Task speakers (10 sesi x 10-way) | 100 |
| &nbsp;&nbsp;&nbsp;&nbsp;- Calibration impostor pool | 1888 |

## Catatan verifikasi vs Tabel 4.1 (F1-03)

- Proposal Tabel 4.1: VoxCeleb1 = 1.251 speaker, VoxCeleb2 = 6.112 speaker (total 7.363).
- Metadata resmi terkini (vox1_meta.csv / vox2_meta.csv, diunduh langsung dari VGG): VoxCeleb1 = 1251 speaker, VoxCeleb2 = 6114 speaker (total 7365).
- VoxCeleb1 cocok persis (1.251). VoxCeleb2 selisih kecil (6114 vs 6.112, ~+2) karena file metadata resmi telah diperbarui sejak angka Tabel 4.1 dikutip dari Nagrani et al. (2018); selisih ini < 0,05% dan tidak memengaruhi validitas skema split (rasio 70/30/10 dihitung dari jumlah speaker aktual, bukan angka historis).

## Guarantee

Disjointness across base_train / reserved_unknown_pool / task_speakers / calibration_impostor_pool is asserted programmatically in `src/data/splits.py::run_full_split` (raises `SpeakerOverlapError` on violation) and re-verified by `tests/test_data_splits.py` (F1-10, blocking).
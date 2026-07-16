# Moodle code submission checklist

Course reminder ([Moodle forum](https://moodle25.technion.ac.il/mod/forum/discuss.php?d=37769)):

| Requirement | Status in this package |
|-------------|------------------------|
| Single ZIP, **code only** | Build with `scripts/pack_moodle_zip.sh` (excludes datasets, checkpoints, results) |
| No datasets / large artifacts in ZIP | Excluded by pack script |
| Dataset downloadable | `scripts/download_processed_data.py` + `data_urls.json` (Google Drive) **or** NOAA pipeline in README §5 |
| Reproduce main experiments | README §3 Setup + §6 Training (`exp_coastal`) |
| `requirements.txt` | Present (optional `environment.yml`) |
| GitHub (encouraged) | https://github.com/ShalevShriki/coastal-ais-trajectory-modeling |

## Dataset Drive links (public)

- `train.parquet`: https://drive.google.com/file/d/1Avt0LDK9LAhMmdhULeHwZdKbXXi6-7vy/view?usp=sharing
- `land_grid_us.npz`: https://drive.google.com/file/d/1aXP3c_M4eAN16I5ltPTreEfnbUfC1_s8/view?usp=sharing

These are embedded in `data_urls.example.json` (shipped in the ZIP).

## Before uploading to Moodle

1. Run `bash scripts/pack_moodle_zip.sh`
2. Upload `AmitaiGal_ShalevShiriki_046211_code.zip` under the Moodle project section

Graders then:

```bash
unzip AmitaiGal_ShalevShiriki_046211_code.zip
cd AmitaiGal_ShalevShiriki_046211_code
export PYTHONPATH="$(pwd)"          # parent of proj/
cd proj/project
pip install -r requirements.txt
python scripts/download_processed_data.py
# Exact args for every report experiment:
bash scripts/exp_coastal/reproduce_experiments.sh            # print
bash scripts/exp_coastal/reproduce_experiments.sh --run flat  # or ar18, all, …
# Full list: EXPERIMENTS.md
```

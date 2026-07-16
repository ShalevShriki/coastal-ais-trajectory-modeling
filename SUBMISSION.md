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

## Before uploading to Moodle

1. Upload these two files to Google Drive (Anyone with the link → Viewer):
   - `data/processed/combined_filtered_smart_coastal/train.parquet` (~2.8 GB)
   - `data/processed/land_grid_us.npz` (~14 KB)
2. Copy `data_urls.example.json` → `data_urls.json` and paste the share links.
3. Run `bash scripts/pack_moodle_zip.sh` and upload the ZIP under the Moodle project section.

Graders then:

```bash
unzip AmitaiGal_ShalevShiriki_046211_code.zip
cd AmitaiGal_ShalevShiriki_046211_code
export PYTHONPATH="$(pwd)"          # parent of proj/
cd proj/project
pip install -r requirements.txt
python scripts/download_processed_data.py
# then train, e.g. Flat LSTM / AR 18h — see README §6
```

# Author-only: Moodle packaging checklist

This file is **not** for course graders (excluded from the Moodle ZIP). Graders follow **README.md → “How to reproduce”**.

| Requirement | Status |
|-------------|--------|
| Single ZIP, code only | `bash scripts/pack_moodle_zip.sh` |
| No datasets / checkpoints in ZIP | Pack script excludes them |
| Dataset downloadable | Drive links in `data_urls.example.json` + `scripts/download_processed_data.py` |
| Reproduce instructions | Top of `README.md` + §6.5 |
| `requirements.txt` / `environment.yml` | Present |
| GitHub | https://github.com/ShalevShriki/coastal-ais-trajectory-modeling |

Drive:

- parquet: https://drive.google.com/file/d/1Avt0LDK9LAhMmdhULeHwZdKbXXi6-7vy/view?usp=sharing
- land grid: https://drive.google.com/file/d/1aXP3c_M4eAN16I5ltPTreEfnbUfC1_s8/view?usp=sharing

Before Moodle upload: rebuild ZIP with `bash scripts/pack_moodle_zip.sh`, then upload `AmitaiGal_ShalevShiriki_046211_code.zip`.

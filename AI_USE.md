# AI use log

Record of every material AI contribution to this project, as required by the
assessment. Each entry gives the date, the tool, what was asked, what was kept
or changed, how it was checked, and any AI errors found.

Tool: Claude Code (Anthropic, model Claude Opus 5.5) unless stated otherwise.

---

## 2026-10-24: Repository and data setup

**Task.** Ignore the raw data in git, set up a Python virtual environment, extract
the image subset from the Kaggle zip, and commit.

**What the AI did**
- Added `data/raw/` and `*.zip` to `.gitignore` so images and archives are never
  committed.
- Created a `.venv` virtual environment (Python 3.12.15 via `uv`, since Python
  was not installed system-wide) and installed pandas 3.1.0.
- Identified `Downloads/images.zip` (12.2 GB) as the Kaggle image archive by
  checking its entry count: 106,339 = 81,060 train + 25,279 test images.
- Found and fixed a bug in `scripts/extract_subset.py`: it only matched images
  stored under an `images/` folder inside the zip, but `images.zip` stores them
  at the top level (`<printer_id>/<print_id>/<timestamp>.jpg`), so it would
  have extracted 0 images. The fix strips an `images/` prefix only if one is
  present.
- Ran the script to extract the 6,227 subset images into `data/raw/images/`.

**What was kept / changed.** All of the above was kept. Committed on branch
`chore/ignore-raw-data` and merged to `main`.

**How it was checked**
- Script output reported all 6,227 of 6,227 subset images found, with no
  missing-file warning; `find data/raw -name '*.jpg' | wc -l` gave 6227.
- `git check-ignore -v data/raw/images/022` confirmed the `data/raw/` rule
  applies, and `git ls-files data/raw` listed no tracked files.

**AI errors found.** None in this session's changes. The script bug above was
caught by inspecting the zip's file list before running the extraction.

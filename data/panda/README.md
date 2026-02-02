# PANDA Dataset Labels

## Download Instructions

1. Go to Kaggle: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data

2. Download `train.csv` file

3. Place it in this directory: `/workspace/zhuo/ETC/data/panda/train.csv`

## Or use Kaggle CLI

```bash
# Install kaggle CLI
pip install kaggle

# Configure API (put your kaggle.json in ~/.kaggle/)
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download the data
kaggle competitions download -c prostate-cancer-grade-assessment -f train.csv -p /workspace/zhuo/ETC/data/panda/
unzip /workspace/zhuo/ETC/data/panda/train.csv.zip -d /workspace/zhuo/ETC/data/panda/
```

## Expected CSV Format

The train.csv should have these columns:
- image_id: the slide identifier (e.g., "0005f7aaab2800f6170c399693a96917")
- isup_grade: ISUP grade from 0 to 5

## ISUP Grade Mapping

| Gleason Score | ISUP Grade |
|---------------|------------|
| Negative | 0 |
| 3+3 = 6 | 1 |
| 3+4 = 7 | 2 |
| 4+3 = 7 | 3 |
| 4+4/3+5/5+3 = 8 | 4 |
| 4+5/5+4/5+5 = 9/10 | 5 |


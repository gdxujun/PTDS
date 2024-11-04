## Progressive Two-Stage Decoder Segment Anything Model for Polyp Segmentation
## Environment
This code was implemented with Python 3.11 and PyTorch 2.3.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./data.
2. Download the pre-trained [SAM 2(Segment Anything)](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt).
3. Training:
```bash
python train.py
```
4. Evaluation:
```bash
python test.py 
```

## Train
```bash
python train.py
```

## Test
```bash
python test.py 
```


# yolov5-on-gtsdb


## Prepare data
Use `notebooks/prepare_data.ipynb` to download and prepare data for training/inference. The notebook is self-explanotry.

## Set path
Set data path in `config/gtsdb.yaml` 

## Train
`python train.py --img 640 --batch 32 --epochs 300 --data data/gtsdb.yaml --weights yolov5s.pt --name run_name --noautoanchor --hyp data/hyps/hyp.scratch.yaml`

The training output is stored at `runs/train/run_name`

## Detect
`python detect.py --weights runs/train/size1360_hypfinetune/weights/best.pt --source /home/ubuntu/Shlok/gtsdb/data/valid/images --name detect_name`

The output images are stored at `runs/detect/detect_name`

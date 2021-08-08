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

## Data question
To annonate the new data, I used [labelImg](https://github.com/tzutalin/labelImg). The images which are to be annoated are found, and the xml annotation files are parsed to add to gt.txt in the following notebook:
`notebooks/data_questions.ipynb`

The class mapping, precision/recall computation, false positive/ false negative images are all taken care of in the following script:
`scripts/analyze_predictions.py`
The script takes all the necessary inputs including the class mapping in the following config file:
`scripts/analyze_config.yaml`

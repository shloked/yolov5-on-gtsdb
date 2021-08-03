# yolov5-on-gtsdb


Use notebooks/prepare_data.ipynb to download and prepare data for training/inference
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /home/ubuntu/Shlok/gtsdb/data  # dataset root dir
train: train/images # train images (relative to 'path') 128 images
val: valid/images  # val images (relative to 'path') 128 images
test:  train/images # test images (optional)

# Classes
nc: 43  # number of classes
names: ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 
          'restriction ends 80', 'speed limit 100', 'speed limit 120', 'no overtaking', 'no overtaking', 
          'priority at next intersection', 'priority road', 'give way', 'stop', 'no traffic both ways', 'no trucks', 
          'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road', 'slippery road', 'road narrows', 
          'construction', 'traffic signal', 'pedestrian crossing', 'school crossing', 'cycles crossing', 'snow', 
          'animals', 'restriction ends', 'go right', 'go left', 'go straight', 'go right or straight', 'go left or straight', 
          'keep right', 'keep left', 'roundabout', 'restriction ends', 'restriction ends']  # class names

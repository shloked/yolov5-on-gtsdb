import numpy as np
import torch
import cv2
from PIL import Image
import os
import pickle
from glob import glob
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
import yaml

# Load config file
config = yaml.safe_load(open("scripts/analyze_config.yaml", 'r'))
dataroot = config['dataroot']
imgfolder = config['imgfolder']
CLASS_MAPPING = config['CLASS_MAPPING']


def IoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


def parse_pred(fname):
    
    preds = json.load(open(fname))
    preds = preds['output']['frames']

    detections = {}
    for frame in preds:
        frn = frame['frame_number']
        for det in frame['signs']:
            bbox = det['coordinates']
            bbox_pascal = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] # convert to [x1, y1, x2, y2]
            bbox_pascal = [float(x) for x in bbox_pascal]
            if frn in detections:
                detections[frn].append((det['class'], *bbox_pascal))
            else:
                detections[frn] = [(det['class'], *bbox_pascal)]
    return detections


def parse_gt(fname):
    gtlist = open(fname).read().split("\n")[:-1]
    gtdict = {}
    for line in gtlist:
        if line=="":
            break
        cols = line.split(';')
        fname = cols[0]
        cls_id = int(cols[-1])
        bbox = list(map(lambda x: float(x), cols[1:5]))
        if cls_id in CLASS_MAPPING:
            if fname in gtdict:
                gtdict[fname].append((CLASS_MAPPING[cls_id], *bbox))
            else:
                gtdict[fname] = [(CLASS_MAPPING[cls_id], *bbox)]
        else:
            pass
    return gtdict

def visualize_fpfn(false_hits, false_type):
    if not len(false_hits)>0:
        return
    res = (50,50)
    num_rows = np.sqrt(len(false_hits)/2)
    num_cols = 2*num_rows
    num_rows, num_cols = int(np.ceil(num_rows)), int(np.ceil(num_cols))
    if (num_rows-1)*num_cols>=len(false_hits):
        num_rows-=1
    canvas = np.zeros((num_rows*res[0],num_cols*res[1],3), dtype = np.uint8)
    for i, fp in enumerate(false_hits):
        fname = os.path.join(dataroot, imgfolder, fp[0])
        bbox = [int(x) for x in fp[1][1:]]
        img = cv2.imread(fname)
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        img = cv2.resize(img, (res[0],res[1]), interpolation = cv2.INTER_AREA)
        x = res[0]*(i//num_cols); y = res[1]*(i%num_cols)
        canvas[x:x+res[0], y:y+res[1], :] = img
        # break
    dump_root = os.path.join("scripts", "analyze_results", false_type)
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)
    cv2.imwrite(os.path.join(dump_root, false_hits[0][1][0]+".png"), canvas)


def compute_recall(gtdict, detdict, cls, iou_thresh=0.5):
    total = 0
    hits = 0
    false_negatives = []
    for key in gtdict:
        gts = gtdict[key]
        gts = list(filter(lambda x: x[0]==cls, gts))
        
        if key in detdict:
            dets = detdict[key]
            dets = list(filter(lambda x: x[0]==cls, dets))
        else:
            dets = []

        for gt in gts:
            total+=1
            fn_flag = True
            for det in dets:
                if IoU(gt[1:], det[1:]) >= iou_thresh:
                    hits+=1
                    fn_flag = False
                    break
            if fn_flag:
                false_negatives.append((key, gt))
    try:
        recall = hits/float(total)
    except:
        recall = -1

    visualize_fpfn(false_negatives, false_type="false_negative")
    return recall




def compute_precision(gtdict, detdict, cls, iou_thresh=0.5):
    total = 0
    hits = 0
    false_positives = []
    for key in detdict:
        dets = detdict[key]
        dets = list(filter(lambda x: x[0]==cls, dets))

        if key in gtdict:
            gts = gtdict[key]
            gts = list(filter(lambda x: x[0]==cls, gts))
        else:
            gts = []
        
        for det in dets:
            total+=1
            fp_flag = True
            for gt in gts:
                if IoU(gt[1:], det[1:]) >= iou_thresh:
                    hits+=1
                    fp_flag = False
                    break
            if fp_flag:
                false_positives.append((key, det))
    try:
        precision = hits/float(total)
    except:
        precision = -1

    visualize_fpfn(false_positives, false_type="false_positive")
    
    # print("here")

    return precision
    

def main():
    gtpath = config['gtpath']
    detpath = config['detpath']
    iou_thresh = config['iou_thresh']
    gtdict  = parse_gt(fname = gtpath)
    detdict = parse_pred(fname = detpath)
    all_classes = sorted(list(set(list(map(lambda x: x[1], CLASS_MAPPING.items())))))
    # all_classes = ["RedRoundSign"]
    outfile = open("scripts/analyze_results.csv", 'w')
    outfile.write("{},{},{}\n".format("Class", "Precision", "Recall"))
    for cls in all_classes:
        recall = compute_recall(gtdict, detdict, cls = cls, iou_thresh= iou_thresh)
        precision = compute_precision(gtdict, detdict, cls = cls, iou_thresh=iou_thresh)
        print("recall: {:.2f} | precision: {:.2f} | Class: {}".format(100*recall, 100*precision, cls))
        outfile.write("{},{},{}\n".format(cls, 100*precision, 100*recall))


if __name__=="__main__":
    main()
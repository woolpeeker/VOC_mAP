import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math

import numpy as np

from reader import (
    pascal_voc_reader,
    weixitong_reader,
    yolo_reader,
    coco_pred_json_reader,
    pred_txt_reader
)

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
CLS_NAMES = ['person']

GT_DATA = yolo_reader('../coco2017_person/labels/val2017', CLS_NAMES)
# PRED_DATA = weixitong_reader('./result.log')
# PRED_DATA = coco_pred_json_reader('../yolov3/results.json', CLS_NAMES)
PRED_DATA = pred_txt_reader('../yolov3/v3iny-half-person.txt')

"""
 Check pred and gt keys
"""
def check_file_ids(gt_data, pred_data):
    gt_ids = list(gt_data.keys())
    pred_ids = list(pred_data.keys())
    for k in gt_ids:
        if k not in pred_ids:
            print(f'{k} not in predictions')
    for k in pred_ids:
        if k not in gt_ids:
            print(f'{k} not in ground-truth')

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def preprocess_pred_data(pred_data):
    _pred_data = []
    for file_id, data in pred_data.items():
        for d in data:
            _pred_data.append({
                'file_id': file_id,
                'box': d['box'],
                'label': d['label'],
                'score': d['score']
            })
    _pred_data = sorted(_pred_data, key=lambda x: x['score'], reverse=True)
    return _pred_data

"""
 Calculate the AP for each class
"""
def calculate_map(GT_DATA, PRED_DATA):
    gt_counter_per_class = {k:0 for k in CLS_NAMES}
    for gt_data in GT_DATA.values():
        for d in gt_data:
            gt_counter_per_class[d['label']] += 1
    PRED_DATA = preprocess_pred_data(PRED_DATA)
    sum_AP = 0.0
    ap_dictionary = {}
    for _, class_name in enumerate(CLS_NAMES):
        """
         Assign detection-results to ground-truth objects
        """
        nd = len(PRED_DATA)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        ig = [0] * nd
        for idx, pred in enumerate(PRED_DATA):            
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            file_id = pred['file_id']
            gt = GT_DATA[file_id]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = pred['box']
            for gt_idx, obj in enumerate(gt):
                # look for a class_name match
                if obj['label'] == class_name:
                    bbgt = obj['box']
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = gt_idx

            # assign detection as true positive/don't care/false positive
            
            # set minimum overlap
            if ovmax >= MINOVERLAP:
                if 'used' not in gt[gt_match]:
                    # true positive
                    tp[idx] = 1
                    gt[gt_match]['used'] = True
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
                    # ig[idx] = 1
                    pass
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(ig):
            ig[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = (tp[idx] + ig[idx]) / (fp[idx] + ig[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP "
        _out_idx = np.arange(0, len(prec), len(prec)//10).tolist().append(-1)
        rounded_prec = [ '%.2f' % elem for elem in mprec[::len(mprec)//10] ]
        rounded_prec.append(mprec[-2])
        rounded_rec = [ '%.2f' % elem for elem in mrec[::len(mrec)//10] ]
        rounded_rec.append(mrec[-2])
        print(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec))
        ap_dictionary[class_name] = ap

    print("\n# mAP of all classes")
    mAP = sum_AP / len(CLS_NAMES)
    text = "mAP = {0:.2f}%".format(mAP*100)
    print(text)

if __name__ == '__main__':
    check_file_ids(GT_DATA, PRED_DATA)
    calculate_map(GT_DATA, PRED_DATA)
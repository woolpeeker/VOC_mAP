import numpy as np
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from PIL import Image
import json

def weixitong_reader(txt_file):
    # weixitongsuo output file
    data = {}
    for line in open(txt_file).readlines():
        search_res = re.search("^识别图片:(.*)大小=", line)
        if search_res:
            file_id = Path(search_res.group(1)).stem
            continue
        search_res = re.search(r"box\[\((\d+),(\d+)\),\((\d+),(\d+)\)\]类别=(\w+), 置信度=(\S+)", line)
        if search_res:
            _data = list(search_res.groups())
            x0 = int(_data[0])
            y0 = int(_data[1])
            x1 = int(_data[2])
            y1 = int(_data[3])
            name = _data[4]
            score = float(_data[5])
            data.setdefault(file_id, []).append({
                'label': name,
                'score': score,
                'box': [x0, y0, x1, y1]
            })
    return data

def pascal_voc_reader(xml_dir):
    data = {}
    for xml_file in Path(xml_dir).glob('*.xml'):
        file_id = xml_file.stem
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            name = obj.find('name').text
            if int(difficult)==1:
                continue
            xmlbox = obj.find('bndbox')
            b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
            data.setdefault(file_id, []).append({
                'label': name,
                'box': b
            })
    return data

def yolo_reader(txt_dir, cnames):
    img_dir = Path(str(txt_dir).replace('labels', 'images'))
    data = {}
    for txt_file in Path(txt_dir).glob('*.txt'):
        file_id = txt_file.stem
        img_file = img_dir / (file_id + '.jpg')
        img = Image.open(str(img_file))
        im_w, im_h = img.size
        for r in np.loadtxt(str(txt_file), dtype=str).reshape([-1, 5]):
            name = cnames[int(r[0])]
            cx = float(r[1]) * im_w
            cy = float(r[2]) * im_h
            w = float(r[3]) * im_w
            h = float(r[4]) * im_h
            x0, y0 = cx - w/2, cy - h/2
            x1, y1 = x0 + w, y0 + h
            data.setdefault(file_id, []).append({
                'label': name,
                'box': [x0, y0, x1, y1]
            })
    return data

def coco_pred_json_reader(json_file, cnames):
    jdata = json.load(open(json_file))
    out_data = {}
    for jdict in jdata:
        file_id = jdict['image_id']
        d = {
            'label': cnames[jdict['category_id']-1],
            'box': jdict['bbox'],
            'score': jdict['score']
        }
        out_data.setdefault(file_id, []).append(d)
    return out_data

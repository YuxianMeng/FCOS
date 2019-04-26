# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: stamp2voc
@time: 2019/4/24 20:38

    将junxu提供的标注转换为标准VOC格式
"""

import json
import os
import sys
import xml.dom.minidom
from random import shuffle
from typing import List, Union

from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def convert(srcs: Union[List[str], str] = "/data/nfsdata/data/yuxian/datasets/person_stamp/detection",
            tgt: str = "/data/nfsdata/data/yuxian/datasets/person_stamp/detection", ):
    """将junxu提供的标注转换为VOC格式"""
    if isinstance(srcs, str):
        srcs = [srcs]
    image_paths = []
    label_paths = []
    for root in srcs:
        image_paths.extend([p.path for p in os.scandir(os.path.join(root, 'images'))])
        label_paths.extend([p.path for p in os.scandir(os.path.join(root, 'labels'))])

    # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
    # self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
    # self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
    new_img_dir = os.path.join(tgt, "JPEGImages")
    new_ann_dir = os.path.join(tgt, "Annotations")
    new_imgset_dir = os.path.join(tgt, "ImageSets", "Main")

    for sub_dir in (new_img_dir, new_ann_dir, new_imgset_dir):
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    for image_path in image_paths:
        label_path = image_path.replace("images", "labels").replace("png", "json")
        print(image_path)
        try:
            junxu_label = json.load(open(label_path, 'r', encoding="utf-8"))["stamps"]
        except FileNotFoundError:
            junxu_label = []
        img = Image.open(image_path).convert("RGB")
        img_new_path = os.path.join(new_img_dir, image_path.split('/')[-1]).replace("png", "jpg")
        w, h = img.size
        # c = img.channel

        new_label = ET.Element("annotation")
        ET.SubElement(new_label, "folder").text = "stamp_0416"
        ET.SubElement(new_label, "filename").text = image_path.split('/')[-1]
        ET.SubElement(new_label, "segmented").text = "0"

        size = ET.SubElement(new_label, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "1"  # todo

        # todo(yuxian): 理论上全空的图片也能提供信息
        if len(junxu_label) == 0:
            continue
        not_recognizable = False
        for label in junxu_label:
            x0, y0, x1, y1, cls, text = int(label["x0"]), int(label["y0"]), int(label["x1"]), int(label["y1"]), label[
                "type"], label["text"]
            if cls == "not_recognizable":
                not_recognizable = True
                break
            object = ET.SubElement(new_label, "object")
            ET.SubElement(object, "name").text = cls
            ET.SubElement(object, "truncated").text = "0"
            ET.SubElement(object, "difficult").text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x0)
            ET.SubElement(bndbox, "xmax").text = str(x1)
            ET.SubElement(bndbox, "ymin").text = str(y0)
            ET.SubElement(bndbox, "ymax").text = str(y1)
        if not_recognizable:
            continue

        new_label_path = os.path.join(new_ann_dir, label_path.split("/")[-1].replace("json", "xml"))

        # tree = ET.ElementTree(new_label)
        # tree.write(new_label_path, encoding="utf-8")
        new_label = ET.tostring(new_label, encoding="utf-8")
        xml_p = xml.dom.minidom.parseString(new_label)
        new_label = xml_p.toprettyxml()

        with open(new_label_path, 'w') as fout:
            fout.write(new_label)
        img.save(img_new_path, "JPEG", quality=80, optimize=True, progressive=True)
        # visual = ET.parse(open(new_label_path, 'rb'))
        # print(visual)


def split(root: str = "/data/nfsdata/data/yuxian/datasets/stamps_voc"):
    """按照VOC的格式切分数据集并生成相应txt"""
    img_dir = os.path.join(root, "JPEGImages")
    # ann_dir = os.path.join(root, "Annotations")
    imgset_dir = os.path.join(root, "ImageSets", "Main")
    img_paths = os.listdir(img_dir)
    shuffle(img_paths)

    train_ratio = 0.8
    train_num = int(len(img_paths) * train_ratio)
    train_paths = img_paths[:train_num]
    test_paths = img_paths[train_num:]

    train_path = os.path.join(imgset_dir, "train.txt")
    test_path = os.path.join(imgset_dir, "test.txt")

    with open(train_path, "w") as fout:
        fout.write("\n".join(t.replace(".jpg", "") for t in train_paths))

    with open(test_path, "w") as fout:
        fout.write("\n".join(t.replace(".jpg", "") for t in test_paths))


if __name__ == '__main__':
    convert(srcs=["/data/nfsdata/data/yuxian/datasets/stamps_20190416"],
            tgt="/data/nfsdata/data/yuxian/datasets/stamps_voc_clean")

    split(root="/data/nfsdata/data/yuxian/datasets/stamps_voc_clean")

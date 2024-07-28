#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import json
import os
import numpy as np
import sys
from detectron2.evaluation import coco_evaluation as coco_eval
PERSON_CLASSES = ['background', 'person']
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.structures import Instances, Boxes
import torch

_PREDEFINED_SPLITS_CROWD_HUMAN=[
    ("mineapple_train", 'detection/train/mineapple_train.json', "CrowdHuman/annotation_train.odgt",'detection/train/images'),
    ("mineapple_test", 'test_data/segmentation/mineapple_test.json', "CrowdHuman/annotation_train.odgt",'test_data/segmentation/images'),
 ]

CROWDHUMAN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "apple"},

]

def get_metadata_for_crowdhuman(odgt_file):
    thing_ids = [k["id"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    
    ret = {
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "odgt_file": odgt_file,
    }
    return ret
def register_mineapple(root):
    for key, json_file , odgt_file, image_root in _PREDEFINED_SPLITS_CROWD_HUMAN:
        register_coco_instances(
            key,
            get_metadata_for_crowdhuman( os.path.join(root,odgt_file)),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
register_mineapple('datasets/mineapple/')
if __name__ == "__main__":
    pred_path = sys.argv[1]
    pred_js = json.load(open(pred_path))
    save_file = f'record.txt'
    evaluator = coco_eval.COCOEvaluator('mineapple_test', output_dir = 'mineapple')
    evaluator.reset()
    for pred_instance in pred_js:
        inputs = [{'image_id': pred_instance['image_id']}]

        instance = Instances((200,200))
        instance.pred_boxes = Boxes(torch.tensor(pred_instance['boxes']))
        instance.scores = torch.tensor(pred_instance['scores'])
        instance.pred_classes = torch.zeros_like( instance.scores, dtype=torch.int32)
        outputs = [{'instances':instance}]
        
        evaluator.process(inputs, outputs)
    print(evaluator.evaluate())
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
import cv2
import tqdm

from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

def create_instances(id,  image_size, predictions):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = np.ones((len(score), 1))
    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument('--image_dir', type=str, default='datasets/CrowdHuman/Images')
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output, exist_ok=True)
    with PathManager.open(args.input, "r") as f:
        js_content = json.load(f)

    for dic in tqdm.tqdm(js_content['images']):
        file_path = os.path.join(args.image_dir, dic["file_name"])
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        predictions = create_instances(dic["id"], img.shape[:2], [item for item in js_content['annotations'] if item['image_id'] == dic['id']])
        vis = Visualizer(img, None)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()
        cv2.imwrite(os.path.join(args.output, basename), vis_pred)

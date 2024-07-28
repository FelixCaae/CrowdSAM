from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse
#This script converts the generated prediction result(in list) into coco annotation format.
#Arg 1: det result in json format
#Arg 2: reference gt label 
#Arg 3: output path
parser = argparse.ArgumentParser(description="This script converts the generated prediction result(in list) into coco annotation format.")
parser.add_argument('-d', '--det_path', type=str)
parser.add_argument('-g', '--gt_path', type=str, default="")
args= parser.parse_args()
val_json = args.gt_path
pred_json = args.det_path
# val_json = 'datasets/coco_split/instances_minival.json'
# val_json = args.'datasets/coco/annotations/instances_val2017.json'
# pred_json = 'test.json'
# Initialize COCO ground truth
def validate(annotation_file, prediction_file):
    
    # coco_categories = self.coco.loadCats(self.coco.getCatIds())
    # coco_category_ids = sorted([category['id'] for category in coco_categories])
    # category_mapping = {coco_id: idx for idx, coco_id in enumerate(coco_category_ids)}
    coco_gt = COCO(annotation_file)  # annotation_file is the path to COCO annotations file
    coco_categories = coco_gt.loadCats(coco_gt.getCatIds())
    coco_category_ids = sorted([category['id'] for category in coco_categories])
    category_inv_mapping = {idx: coco_id for idx, coco_id in enumerate(coco_category_ids)}
    
    img_ids = coco_gt.getImgIds()
    
    # Load your predictions
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    orig_pred = []
    for pred in predictions:
        pred['category_id'] = category_inv_mapping[pred['category_id']]
    # Initialize COCO detections
    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# validate(val_json, pred_val_json)
# validate(mval_json, pred_mval_json)
validate(val_json, pred_json)

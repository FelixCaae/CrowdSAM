def do_eval_coco(image_ids, coco, results, flag):
    from pycocotools.cocoeval import COCOeval
    assert flag in ['bbox', 'segm', 'keypoints']
    # Evaluate
    coco_results = coco.loadRes(results)
    cocoEval = COCOeval(coco, coco_results, flag)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize() 
    return cocoEval


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
val_json = 'datasets/OCHuman/ochuman_coco_format_val_range_0.00_1.00.json'
mval_json = 'datasets/OCHuman/ochuman_coco_format_val_range_0.50_0.75.json'
hval_json = 'datasets/OCHuman/ochuman_coco_format_val_range_0.75_1.00.json'
pred_val_json = 'occ_val_00_10.json'
pred_mval_json = 'occ_val_50_75.json'
pred_hval_json = 'occ_val_75_100.json'
# Initialize COCO ground truth

def validate(annotation_file, prediction_file):
    coco_gt = COCO(annotation_file)  # annotation_file is the path to COCO annotations file
    img_ids = coco_gt.getImgIds()
    
    # Load your predictions
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    # Initialize COCO detections
    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# validate(val_json, pred_val_json)
# validate(mval_json, pred_mval_json)
validate(hval_json, pred_hval_json)

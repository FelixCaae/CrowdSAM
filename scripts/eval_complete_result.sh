annot_file='midval_visible.json'
python tools/convert2coco.py -d final.json   -o test.json
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/annotation_val.odgt --remove_empty_gt
rm test.json
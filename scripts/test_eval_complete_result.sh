annot_file='midval_visible_2.json'
echo "test evaluation with int id mode"
python tools/convert2coco.py -d final.json   -o test.json -g datasets/CrowdHuman/$annot_file --ref_img_id_type int
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/$annot_file
rm test.json

echo "test evaluation with str id mode with remove empty_gt"
python tools/convert2coco.py -d final.json   -o test.json -g datasets/CrowdHuman/$annot_file --ref_img_id_type str
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/annotation_val.odgt --remove_empty_gt --visible_flag
rm test.json
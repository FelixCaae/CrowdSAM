num_imgs=$1
num_nodes=$2
batch_size=$(($1 / $2))
output_file="final.json"
annot_file="datasets/CrowdHuman/midval_visible_100.json"
odgt_file="annotation_val.odgt"
# annot_file="train_visible.json"  # "train_visible" #
exec_file="crowdhuman-main.py"
options="--label_file $annot_file  --dataset crowdhuman --mask_selection max_area  --ref_feature_path template_features/ref_feature_50.pkl  --num_prompts 400 --crop_n_layers 0 --box_nms_thresh 0.65 --pred_iou_thresh 0.2 --min_mask_region_area 100 --crop_n_layers 0 --dino_model dinov2_vits14" 
# options="$options --sam_model vit_l --sam_adapter_checkpoint ./sam_adapter_weights/10_shot.pth  --mask_selection max_iou" # 
options="$options --sam_model mobilevit_t --sam_adapter_checkpoint ./sam_adapter_weights/mobile_10_shot.pth  --mask_selection max_iou --sam_checkpoint mobile_sam/mobile_sam.pt" # --apply_box_offsets

for ((i=0; $i< $num_nodes; i++))
do
    start_idx=$(($i*$batch_size))
    end_idx=$((($i+1)*batch_size))
    save_dir="batch_run_$i"
    echo "$start_idx, $end_idx, $save_dir"
    srun -c 4 --mem 40G --gres=gpu:1 python $exec_file  --output_dir vis_output/$save_dir $options --start_idx $start_idx --num_imgs $end_idx &
done
wait
# touch $output_file
json_list=""
for ((i=0; $i< $num_nodes; i++))
do
    save_dir="batch_run_$i"
    json_list="$json_list vis_output/$save_dir/result.json"
done 
echo "Merge results."
python tools/merge_json.py final.json $json_list
echo "Testing with visible flag"
python tools/convert2coco.py -d final.json   -o test.json -g $annot_file --ref_img_id_type str
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman//$odgt_file --remove_empty_gt --visible_flag
# echo "Testing without visible flag"
# python tools/convert2coco.py -d final.json   -o test.json -g datasets/CrowdHuman/$annot_file --ref_img_id_type str --full_box
# python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/$odgt_file --remove_empty_gt 

rm test.json
echo "all processes done"

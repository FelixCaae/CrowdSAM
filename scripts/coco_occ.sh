num_imgs=$1
num_nodes=$2
batch_size=$(($1 / $2))
output_file="final.json"
annot_file="/irip/caizhi_2019/label_completer/datasets/occ_coco/instances_occ2017_new.json"
# annot_file="train_visible.json"  # "train_visible" #
exec_file="crowdhuman-main.py"
options="--label_file $annot_file  --mask_selection max_area  --ref_feature_path template_features/ref_feature_50.pkl  --num_prompts 400 --box_nms_thresh 0.65 --pred_iou_thresh 0.1 --min_mask_region_area 100 --crop_n_layers 1 --dataset coco_occ" 
# options="$options --sam_model vit_l --sam_adapter_checkpoint ./sam_adapter_weights/10_shot.pth  --mask_selection max_iou" # 
options="$options --sam_model adaptvit_l --sam_adapter_checkpoint ./sam_adapter_weights/600_coco_shot.pth  --mask_selection max_iou" # --apply_box_offsets

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
python tools/convert2coco_list.py -d final.json  -o test.json 
python tools/coco_eval.py -d test.json -g $annot_file
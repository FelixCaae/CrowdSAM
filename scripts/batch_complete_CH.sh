num_imgs=$1
num_nodes=$2
batch_size=$(($1 / $2))
output_file="final.json"
annot_file="midval_visible.json"
# options="--label_file $annot_file  --select_biggest_mask  --ref_feature_path ref_feature_test.pkl --ref_feature_fusion mean  --num_prompts 10 50 200 400  --score_thresh -1 --neg_sim_thresh 0  --focus_on_fg --pos_sim_thresh 0.15  --twostage_prompt --max_size 1536"
options="--label_file $annot_file  --select_biggest_mask  --ref_feature_path ref_feature_50.pkl  --num_prompts 10 50 100 400 --score_thresh -1 --neg_sim_thresh 0  --focus_on_fg --pos_sim_thresh 0.15  --max_size 1024 --iou_thresh 0.8 --ref_feature_fusion weighted "
# options="--label_file $annot_file  --select_biggest_mask  --ref_feature_path ref_feature_test.pkl --ref_feature_fusion mean  --num_prompts 200 --score_thresh -1 --focus_on_fg  --neg_sim_thresh 0 --feat_size 80 --multiscale"
for ((i=0; $i<$num_nodes; i++))
do
    start_idx=$(($i*$batch_size))
    end_idx=$((($i+1)*batch_size))
    save_dir="batch_run_$i"
    echo "$start_idx, $end_idx, $save_dir"
    srun -c 4 --mem 40G --gres=gpu:1 python main.py  --output_dir vis_output/$save_dir $options --start_idx $start_idx --num_imgs $end_idx &
done
wait
# touch $output_file
json_list=""
for ((i=0; $i< $num_nodes; i++))
do
    save_dir="batch_run_$i"
    json_list="$json_list vis_output/$save_dir/result.json"
done 
python tools/merge_json.py final.json $json_list
bash scripts/eval_complete_result.sh final.json datasets/CrowdHuman/$annot_file
echo "all processes done"

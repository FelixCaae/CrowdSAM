num_imgs=$1
num_nodes=$2
batch_size=$(($1 / $2))
output_file="final.json"
annot_file="midval_visible_100.json"
# annot_file="train_visible.json"  # "train_visible" #
exec_file="crowdhuman-main_v2.py"
options="--label_file $annot_file --crop_n_layer 2"
echo 'excuting batch processing for sam automatic_generator baselines' 
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
python tools/merge_json.py final.json $json_list
python tools/convert2coco.py -d final.json   -o test.json -g datasets/CrowdHuman/$annot_file --ref_img_id_type str
echo "Testing with visible flag"
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/annotation_val.odgt --remove_empty_gt --visible_flag

echo "Testing without visible flag"
python tools/crowdhuman_eval.py -d test.json -g datasets/CrowdHuman/annotation_val.odgt --remove_empty_gt 

rm test.json
echo "all processes done"

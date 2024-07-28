
output_file="final.json"
annot_file="midval_visible.json"
save_dir='visualize_adapter_apply_box_offsets'
# options="--label_file $annot_file  --select_biggest_mask  --ref_feature_path ref_feature_test.pkl --ref_feature_fusion mean  --num_prompts 10 50 200 400  --score_thresh -1 --neg_sim_thresh 0  --focus_on_fg --pos_sim_thresh 0.15  --twostage_prompt --max_size 1536"
options="--label_file $annot_file  --mask_selection max_area  --ref_feature_path template_features/ref_feature_50.pkl  --num_prompts 10 50 100 400 --score_thresh -1 --focus_on_fg --pos_sim_thresh 0.15  --iou_thresh 0.8 --ref_feature_fusion weighted  --score_thresh 0.3 --twostage_prompt"

#adapter option
options="$options --sam_model adaptvit_l --sam_adapter_checkpoint ../Personalize-SAM/tuned_mask_decoder_for_visible_CH.pth  --mask_selection max_iou --apply_box_offsets"

# options="--label_file $annot_file  --select_biggest_mask  --ref_feature_path ref_feature_test.pkl --ref_feature_fusion mean  --num_prompts 200 --score_thresh -1 --focus_on_fg  --neg_sim_thresh 0 --feat_size 80 --multiscale"

srun -c 4 --mem 40G --gres=gpu:1 python main.py  --output_dir vis_output/$save_dir $options --visualize

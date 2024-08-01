import os
import subprocess
import concurrent.futures
import argparse
import json
import yaml
import crowdsam.utils as utils
def run_script(start_idx, end_idx, rank, exec_file,config_file):
    cmd = [
        
        # 'srun', '-c', '4', '--mem', '40G', '--gres=gpu:1', 
        'python', exec_file, 
        '--config_file', config_file,
        '--save_path', f'temp_result_{rank}.json', 
        '--start_idx', str(start_idx), 
        '--end_idx', str(end_idx),
        '--local_rank', str(rank),
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def merge_json(json_files):    # Initialize an empty list to hold merged data
    merged_data = []
    # Load and merge JSON files
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)
    for json_file in json_files:
        os.remove(json_file)
    # Write merged data to the specified output JSON file
    return merged_data
def convert_to_coco(det_result, gt_js):
    #
    id_ = 0
    annotations = []
    category_id = 1
    
    image_items = gt_js['images']
    categories = gt_js['categories']
    for img_item in image_items:
        img_item['id'] = img_item['file_name'][:-4]

    for k,item in enumerate(det_result):
        #convert image id to integer by defaults
        if image_items != []:
            image_id = image_items[k]['id']
        else:
            image_id = item['image_id']
        scores = item['scores']
        boxes =  item["boxes"] 
        for score,box in zip(scores, boxes):
            area = (box[3] - box[1]) * (box[2] - box[0])
            box [2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            annot = {"category_id":category_id, "bbox":box, "image_id":image_id, "iscrowd":False, "area": area, "id":id_, "score":score}
            id_ += 1
            annotations.append(annot)
    final_result= {"images":image_items, "annotations":annotations, 'categories':categories}
    return final_result

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts concurrently")
    parser.add_argument('-n','--num_nodes', type=int,  default=8, help='Number of nodes to use')
    parser.add_argument('-c','--config_file', default='./configs/crowdhuman.yaml')
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    config = utils.modify_config(config, args.options)
    print(yaml.dump(config, default_flow_style=False, default_style='' ))
    #load yaml
    gt_js = json.load(open(config['data']['json_file']))
    num_imgs = len(gt_js['images'])
    num_nodes = args.num_nodes
    annot_file = config['data']['json_file']
    odgt_file = config['data']['odgt_file']
    exec_file = 'test.py'
    config_file = args.config_file
    options = args.options
    

    # Run the python scripts concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
        futures = []
        batch_size = num_imgs // num_nodes
        for i in range(num_nodes):
            start_idx = i * batch_size
            if i == num_nodes - 1:
                end_idx = num_imgs
            else:
                end_idx = (i + 1) * batch_size 
            futures.append(executor.submit(run_script, start_idx, end_idx, i, exec_file, config_file))
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # Merge JSON results
    json_list = [f"temp_result_{i}.json" for i in range(num_nodes)]
    merged_result = merge_json(json_list)
    
    coco_json = convert_to_coco(merged_result, gt_js)
    json.dump(coco_json, open('test.json','w'), ensure_ascii=True)

    eval_cmd = f"python tools/crowdhuman_eval.py -d test.json -g {odgt_file} --remove_empty_gt --visible_flag"
    print(f"Evaluating with command: {eval_cmd}")
    subprocess.run(eval_cmd, shell=True)
    os.remove("test.json")
    print("All processes done")

if __name__ == "__main__":
    main()
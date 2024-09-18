
import os
import sys
import json
import glob
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import torch
from crowdsam.model import CrowdSAM
from crowdsam.utils import (load_img_and_annotation, setup_logger,
                   data_meta, load_config, modify_config,
                   visualize_result,evaluate_boxes)
import argparse
def envrion_init():
    parser = argparse.ArgumentParser(description="CrowdSAM argparser")
    parser.add_argument('--mode', type=str, choices=['seg', 'bbox'], default='seg')
    #data related
    parser.add_argument('-c','--config_file', type=str, default='./configs/crowdhuman.yaml')
    parser.add_argument('-i', '--input', help="Input of demo, it could be a directory of images, a single image or a glob pattern such as 'test_dir/*.jpg'", type=str)
    parser.add_argument('-o', '--output', help='Output path of demo', type=str, default='demo_out')
    args = parser.parse_args()
    configs = load_config(args.config_file)
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(args.output + '/log') 
    logger.info(args)
    return args, configs, logger

if __name__ == '__main__':
    #===========>init environments
    args, config, logger = envrion_init()
    n_class, class_names = data_meta[config['data']['dataset']][1:]
    model = CrowdSAM(config, logger)

    #parse the input files
    if os.path.isdir(args.input):
        list_of_files = os.listdir(args.input)
        image_files = [os.path.join(args.input, f) for f in list_of_files]
    elif os.path.exists(args.input):
        image_files = [args.input]
    else:
        image_files =  glob.glob(os.path.expanduser(args.input))
    #===========>run in loop and collect result
    output_content = []
    logger.info(f'total images  to process { len(image_files)}')
    for image_file in tqdm(image_files):
        # load one image
        image = Image.open(image_file)
        result = model.generate(image)
        instance_dict = {'image_file':image_file}
        instance_dict.update({k:v.tolist() for k,v in result.items() if k in ['boxes', 'scores', 'categories'] })
        instance_dict.update({k:v for k,v in result.items() if k in ['rles'] })
        output_content.append(instance_dict)
        image_name = image_file.split('/')[-1].split('.')[0]
        save_path = os.path.join(args.output, f'{image_name}.jpg')
        visualize_result(image, result,  class_names, save_path, conf_thresh= config['vis']['vis_thresh'], vis_masks = args.mode=='seg')

    file_path = os.path.join(args.output,'result.json')
    print(f'dump json file to {file_path}')
    json.dump(output_content, open(file_path, 'w'), ensure_ascii=True)
    
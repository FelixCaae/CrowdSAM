
import os
import sys
import json
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
    parser.add_argument('--mode', type=str, choices=['seg', 'ref_only'], default='seg')
    #data related
    parser.add_argument('--start_idx',type=int, default=0)
    parser.add_argument('--end_idx',type=int, default=-1) # -1 represents using all images
    parser.add_argument('-c','--config_file', type=str, default='./configs/crowdhuman.yaml')
    parser.add_argument('-v','--visualize',help='visualize the outputs', action="store_true")
    parser.add_argument('-s','--save_path',help='the path to dump json result', type=str, default="")
    parser.add_argument('-r','--local_rank', type=int, default=0)
    parser.add_argument('options',nargs=argparse.REMAINDER)
    # parser.add_argument('--dataset',type=str, default="crowdhuman")
    args = parser.parse_args()

    configs = load_config(args.config_file)
    configs = modify_config(configs, args.options)
    np.random.seed(configs['environ']['seed'])
    torch.random.manual_seed(configs['environ']['seed'])
    os.makedirs(configs['environ']['output_dir'], exist_ok=True)
    os.makedirs(configs['environ']['output_dir'] + '/log', exist_ok=True)
    logger = setup_logger(configs['environ']['output_dir'] + '/log') 
    logger.info(args)
    return args, configs, logger

if __name__ == '__main__':
    #===========>set arguments or options 
    args, config, logger = envrion_init()
    #==========>load data meta infos
    dataset_path = config['data']['dataset_root']
    n_class, class_names = data_meta[config['data']['dataset']][1:]
    if 'cuda' in config['environ']['device']:
        torch.cuda.set_device(args.local_rank) 
        config['environ']['device'] =  f'cuda:{args.local_rank}'
    #===========>configure model
    model = CrowdSAM(config, logger)
    # annot_path = os.path.join(dataset_path[args.dataset], args.label_file)
    annot_path = config['data']['json_file']
    #===========> A simple data loading strategy that can be replaced with a DDP dataloader in the future update.
    logger.info('load images and annotations from crowdhuman dataset..')
    annots = json.load(open(annot_path))
    if args.end_idx == -1:
        end_idx = len(annots['images'])
    else:
        end_idx = min(args.end_idx, len(annots['images']))
    image_ids = [ i for i in range(args.start_idx,end_idx)]
    #===========>run in loop and collect result
    output_content = []
    logger.info(f'total images  to process { len(image_ids)}')
    for id_ in tqdm(image_ids):
        logger.debug(f'start processing {id_}')
        # load one image
        image, gt_boxes, image_id = load_img_and_annotation(dataset_path, annots, config['data']['dataset'], id_)
        result = model.generate(image)
        # AP, recall, = round(AP, 3), round(recall, 3)
        # FP_ind, FN_ind = None, None
        instance_dict = {'image_id':image_id,  'num_gt':len(gt_boxes)-1}
        # instance_dict.update({'AP':AP, 'Recall':recall, })    
        instance_dict.update({k:v.tolist() for k,v in result.items() if k in ['boxes', 'scores', 'categories'] })
        instance_dict.update({k:v for k,v in result.items() if k in ['segmentations'] })
        # save detection results in json 
        output_content.append(instance_dict)
        logger.debug(f'process for image:{id_} is done')
        # visualize the detected objects
        if args.visualize:
            save_path = os.path.join(config['environ']['output_dir'], f'{id_}.jpg')
            result['gt_boxes'] = gt_boxes
            FP_list, FN_list = evaluate_boxes(result['boxes'], result['scores'], gt_boxes, 0.5)[2:]
            visualize_result(image, result,  class_names, save_path, conf_thresh= config['vis']['vis_thresh'], FP_ind=FP_list, FN_ind=FN_list)#,  FP_ind = FP_ind, FN_ind = FN_ind)
        del result        

    if args.save_path == "":
        file_path = os.path.join(config['environ']['output_dir'],'result.json')
        print(f'dump json file to {file_path}')
        json.dump(output_content, open(file_path, 'w'), ensure_ascii=True)
    else:
        json.dump(output_content, open(args.save_path, 'w'), ensure_ascii=True)
    
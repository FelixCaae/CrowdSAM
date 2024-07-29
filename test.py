
import os
import sys
import json
from tqdm import tqdm
from crowdsam import CrowdSAM
from utils import (load_img_and_annotation,
                   data_meta, envrion_init,
                   visualize_result,evaluate_boxes)

if __name__ == '__main__':
    #===========>set arguments or options 
    args, config, logger = envrion_init()
    #==========>load data meta infos
    dataset_path = config['data']['dataset_root']
    n_class, class_names = data_meta[config['data']['dataset']][1:]
    #===========>configure model
    lc = CrowdSAM(config,logger)
    output_file = 'result.json'
    # annot_path = os.path.join(dataset_path[args.dataset], args.label_file)
    annot_path = config['data']['json_file']
    #===========> A simple data loading strategy that can be replaced with a DDP dataloader in the future update.
    logger.info('load images and annotations from crowdhuman dataset..')
    annots = json.load(open(annot_path))
    if args.num_imgs == -1:
        num_img = len(annots['images'])
    else:
        num_img = min(args.num_imgs, len(annots['images']))
    image_ids = [ i for i in range(args.start_idx,num_img)]
    #===========>run in loop and collect result
    output_content = []
    logger.info(f'total images  to process { len(image_ids)}')
    for id_ in tqdm(image_ids):
        logger.debug(f'start processing {id_}')
        # load one image
        image, gt_boxes, image_id = load_img_and_annotation(dataset_path, annots, config['data']['dataset'], id_)
        result = lc.generate(image)
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
        
    file_path = os.path.join(config['environ']['output_dir'], output_file)
    json.dump(output_content, open(file_path, 'w'), ensure_ascii=True)
    
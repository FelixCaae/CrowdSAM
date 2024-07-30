import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from label_completer import LabelCompleter
from PIL import Image
import cv2
from utils import ( setup_logger, 
                   draw_box, draw_mask, draw_point)


parser = argparse.ArgumentParser(description="")
parser.add_argument('--image_dir',type=str, default="demo")
parser.add_argument('--dino_repo',type=str, default='../dinov2')
parser.add_argument('--sam_checkpoint',type=str, default='../segment-anything/sam_vit_l_0b3195.pth')
parser.add_argument('--sam_model',type=str, default='vit_l')
parser.add_argument('--ref_feature_path',type=str, default='ref_feature_test.pkl')
parser.add_argument('--ref_feature_fusion',type=str, default='mean')
parser.add_argument('--twostage_prompt',action="store_true")
parser.add_argument('--focus_on_fg', action='store_true')

parser.add_argument('--num_points',type=int, default=200)
parser.add_argument('--num_prompts',type=int, default=50)
parser.add_argument('--num_neg_points',type=int, default=100)
parser.add_argument('--num_neg_prompts',type=int, default=10)
parser.add_argument('--pos_sim_thresh',type=float, default=0.2)
parser.add_argument('--neg_sim_thresh',type=float, default=0.05)
parser.add_argument('--twostage_segment', action="store_true")

#post processing parameters
parser.add_argument('--output_dir',type=str, default='vis_output/demo')
parser.add_argument('--num_vis_box',type=int, default=40)

if __name__ == '__main__':
    #===========>fix seed for reproduction
    np.random.seed(42)
    torch.random.manual_seed(42)
    #===========>set arguments or options 
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir) 
    logger.info(args)
    # dataset_crowdhuman = ("datasets/CrowdHuman", "", "val.json")
    # dataset_coco = ("datasets/coco", "annotations", "instances_train2017.json")
    #===========>load annotations 
    logger.info('load images and annotations from crowdhuman dataset..')
    #===========>set ref feature to None in generateing ref mode
    ref_feature_path =  args.ref_feature_path
    #===========>load model
    lc = LabelCompleter(args.sam_model, args.sam_checkpoint, args.dino_repo,
                        mode = "seg", feat_size=80,
                        ref_feature_path= ref_feature_path,
                        ref_feature_fusion = "mean", 
                        twostage_prompt = True,
                        focus_on_fg = True,
                        num_points= args.num_points,num_prompts= args.num_prompts,
                        num_neg_points= args.num_neg_points, num_neg_prompts= args.num_neg_prompts,
                        pos_sim_thresh=args.pos_sim_thresh, neg_sim_thresh=args.neg_sim_thresh,
                        select_biggest_mask = True,
                        merge_masks=True,
                        twostage_segment=True,
                        twostage_postprocess  = True, score_thresh=-1,
                        visualize= True,beta = 0, output_dir= args.output_dir, logger=logger)

    #===========>run in loop and collect result
    image_names = os.listdir(args.image_dir)
    result_json = []
    logger.info(f'total images  to process { len(image_names)}')
    for name_ in tqdm(image_names):
        
        logger.info(f'start processing {name_}')
        # load one image 
        image = Image.open(os.path.join(args.image_dir, name_))
        lc.set_image(image, )
        result = lc.process()
        # when mode == seg or prompt we collect some metrics like recall and ap
        logger.info('start evalulating..')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for mask in  result['masks']:
            image= draw_mask(np.array(mask), image, random_color=True)
        for score,box in zip(result['scores'],result['boxes']):
            image = draw_box(box,image, 'green',  float(score))
        for point in result['prompts']:
            image = draw_point(point, image)
        save_path = f'{args.output_dir}/{name_}'
        print(save_path)
        cv2.imwrite(save_path,image)
 
    #============> out loop here
    ##save referenion results

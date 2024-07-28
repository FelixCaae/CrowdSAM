import json
import os
import argparse
#This script converts the generated prediction result(in list) into coco annotation format.
#Arg 1: det result in json format
#Arg 2: reference gt label 
#Arg 3: output path
parser = argparse.ArgumentParser(description="This script converts the generated prediction result(in list) into coco annotation format.")
parser.add_argument('-d', '--det_path', type=str)
parser.add_argument('-g', '--ref_path', type=str, default="")
parser.add_argument('-r', '--ref_img_id_type', type=str, default='str')
parser.add_argument('-o', '--output_path', type=str, default="./output.json")
parser.add_argument('-f', '--full_box', action='store_true')

def convert_to_coco(det_result, ref_image_items=[], full_box=False):
    id_ = 0
    annotations = []
    for k,item in enumerate(det_result):
        #convert image id to integer by defaults
        if ref_image_items != []:
            image_id = ref_image_items[k]['id']
        else:
            image_id = item['image_id']
        scores = item['scores']
        boxes =  item["boxes"] if not full_box else item['fboxes']
        segmentations = item['segmentations']
        try:
            categories = item['categories']
        except:
            continue
        for class_id, score,box,seg in zip(categories, scores, boxes,segmentations):
            area = (box[3] - box[1]) * (box[2] - box[0])
            box [2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            annot = {"category_id":class_id, "bbox":box, "image_id":image_id, "iscrowd":False, "area": area, "id":id_, "score":score,'segmentation':seg}
            id_ += 1
            annotations.append(annot)
    return annotations

if __name__ == '__main__':
    args = parser.parse_args()
    det_path = args.det_path #
    gt_path = args.ref_path

    det_result = json.load(open(det_path))    
    #sort det by image id
    det_result = sorted(det_result, key=lambda x:int(x['image_id']))
    if os.path.exists(gt_path):
        gt_result = json.load(open(gt_path))
        image_items = gt_result['images']
        categories = gt_result['categories']
        if args.ref_img_id_type == 'str':
            for img_item in image_items:
                img_item['id'] = img_item['file_name'][:-4]
        elif args.ref_img_id_type == 'int':
            for img_item in image_items:
                img_item['id'] = img_item['id']            
        #make sure det result has the same image set with ref gt label
        if len(image_items) != len(set(item['image_id'] for item in det_result)):
            print('warnings: image_items is not as long as det results')
    else:
        image_items = []
        categories = {'person':0}
    #start converting        
    annotations = convert_to_coco(det_result, ref_image_items = image_items, full_box=args.full_box)
    
    #if gt_ref is not provided, we make an
    if image_items is []:
        image_ids = set([item['image_id'] for item in annotations])
        image_items = []
    final_result= annotations
    json.dump(final_result, open(args.output_path, 'w'), ensure_ascii=True)

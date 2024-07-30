import argparse
import json
import os

import cv2 as cv
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", '--data-path',
        default="dataset/crowdhuman",
        type=str,
        help='the path of CrowdHuman dataset'
    )
    parser.add_argument(
        "-o", "--odgt-path",
        default='annotation_val.odgt',
        type=str,
        help="the path of CrowdHuman odgt file"
    )
    parser.add_argument(
        "-s", "--save-path",
        default='val.json',
        type=str,
        help='the path to save json file'
    )
    parser.add_argument(
        "-v", "--visible",
        action='store_true',
        help="keep visible box",
    )
    parser.add_argument(
        "-f", "--full",
        default=1,
        type=int,
        help="keep full box",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="keep head box",
    )
    parser.add_argument(
        "--rm-occ",
        default=1,
        type=int,
        help="remove occluded box",
    )
    parser.add_argument(
        "--rm-hignore",
        default=0,
        type=int,
        help="remove ignored head box",
    )
    parser.add_argument(
        "--rm-hocc",
        default=0,
        type=int,
        help="remove occluded head head",
    )
    parser.add_argument(
        "--size",
        default=999999999,
        type=int,
        help="remove occluded head head",
    )
    parser.add_argument(
        "--rm-hunsure",
        default=0,
        type=int,
        help="keep unsure head box",
    )
    parser.add_argument(
        "--tag",
        default="person",
        type=str,
        help="keep box with tag 'person' or 'mask' or 'both'"
    )
    parser.add_argument(
        "--shuffle",
        action='store_true',
        help="keep unsure head box",
    )
    return parser


def readlines(filename):
    print("start read odgt file ")
    with open(filename, 'r') as f:
        lines = f.readlines()
    name = filename.split(os.sep)[-1].split('.')[0].split('_')[-1]
    print(f"{len(lines)} images in CrowdHuman {name} dataset")

    return [json.loads(line.strip('\n')) for line in lines]


def crowdhuman2coco(args, odgt_path, json_path, data_path):
    records = readlines(odgt_path)  # A list contains dicts
    json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }  # coco format
    bbox_id = 1
    categories = {}
    print("start convert")
    import tqdm
    if len(records) == 1:
        records = records[0]
        
    ids = np.arange(len(records))
    
    if args.shuffle:
        np.random.seed(42)
        records = np.array(records)
        rand_ind = np.random.choice(np.arange(len(ids)), len(ids), replace=False)
        # records = records[rand_ind]
        ids = ids[rand_ind]
    for image_id, item_id in tqdm.tqdm(enumerate(ids)):
        image_dict = records[item_id]
        file_name = image_dict['ID'] + '.jpg'
        # print(data_path, file_name)
        im = cv.imread(os.path.join(data_path , file_name), 0)  # gain height and width
        assert im is not None
        image = {
            'file_name': file_name,
            'height': im.shape[0],
            'width': im.shape[1],
            'id': int(image_id),
        }
        # print(file_name)
        json_dict['images'].append(image)
        gt_box = image_dict['gtboxes']  # A list contains dicts

        for _, instance in enumerate(gt_box):
            annotation = {}
            category = instance['tag']

            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id

            if instance['tag'] == args.tag or 'both' == args.tag:
                annotation['category_id'] = categories[category]
            else:
                continue
            # import ipdb;ipdb.set_trace()
            

            if args.full:
                attr = instance['extra']
                # if args.rm_occ and attr['ignore']:
                #     continue
                annotation['bbox'] = instance['fbox']
            if args.visible:
                attr = instance['extra']
                # if args.rm_occ and attr['ignore']:
                #     continue
                annotation['bbox'] = instance['vbox']
                if type(annotation['bbox'][0]) is list:
                    annotation['bbox'] = annotation['bbox'][0]
            if args.head:
                attr = instance['head_attr']
                if args.rm_hocc and attr['occ']:
                    continue
                if args.rm_hunsure and attr['unsure']:
                    continue
                if args.rm_hignore and attr['ignore']:
                    continue
                annotation['hbox'] = instance['hbox']

            annotation['image_id'] = int(image_id)#image_dict['ID']
            annotation['iscrowd'] = False
            area = annotation['bbox'][2] * annotation['bbox'][3]
            annotation['area'] = area
            annotation['id'] = bbox_id
            if 'ignore' in attr:
                annotation['ignore'] = attr['ignore']
            else:
                annotation['ignore'] = 0
            bbox_id += 1
            json_dict['annotations'].append(annotation)
        if image_id == args.size - 1:
            print(args.size, len(json_dict['annotations']))
            break
    for cate, cid in categories.items():
        cat = {
            'supercategory': cate,
            'id': cid,
            'name': cate
        }
        json_dict['categories'].append(cat)

    print("start write json")
    print('total images', len(json_dict['images']))
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print(f"Json file have been dumped to {json_path}")


def main():
    parser = make_parser()
    args = parser.parse_args()
    img_path = os.path.join(args.data_path, 'Images')
    odgt_path = os.path.join(args.data_path, args.odgt_path)
    save_path = os.path.join(args.data_path, args.save_path)
    crowdhuman2coco(args, odgt_path, save_path ,img_path)


if __name__ == "__main__":
    main()

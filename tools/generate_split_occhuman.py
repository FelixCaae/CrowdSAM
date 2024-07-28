from ochumanApi.ochuman import OCHuman
# <Filter>: 
#      None(default): load all. each has a bbox. some instances have keypoint and some have mask annotations.
#            images: 5081, instances: 13360
#     'kpt&segm' or 'segm&kpt': only load instances contained both keypoint and mask annotations (and bbox)
#            images: 4731, instances: 8110
#     'kpt|segm' or 'segm|kpt': load instances contained either keypoint or mask annotations (and bbox)
#            images: 5081, instances: 10375
#     'kpt' or 'segm': load instances contained particular kind of annotations (and bbox)
#            images: 5081/4731, instances: 10375/8110
ochuman = OCHuman(AnnoFile='../datasets/OCHuman/ochuman.json', Filter='kpt&segm')
image_ids = ochuman.getImgIds()
print ('Total images: %d'%len(image_ids))
ochuman.toCocoFormart(subset='val', maxIouRange=(0.5, 0.75), save_dir='../datasets/OCHuman/')
ochuman.toCocoFormart(subset='val', maxIouRange=(0.75, 1), save_dir='../datasets/OCHuman/')
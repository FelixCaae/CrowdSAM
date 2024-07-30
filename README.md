
## [Crowd-SAM: SAM as a Smart Annotator for Object Detection in Crowded Scenes](https://arxiv.org/abs/2407.11464)
------------

## 1. Introduction
Crowd-SAM is a novel few-shot object detection and segmentation method designed to handle crowded scenes. We combine SAM with the specifically designed efficient prompt  sampler and a mask selection PWD-Net to achieve fast and accurate pedestrian detection! Crowd-SAM achieves 78.4\% AP on the Crowd-Human benchmark with 10 supporting images which is comparable to supervised detectors. 

![PDF Page](figures/fig1.jpg)
## 2. Installation
We recommend to use virtual enviroment, *e.g. Conda*,  for installation:
1. Create virtual environment:
   ```bash
   conda create -n crowdsam python=3.8
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crowd-sam.git
   cd crowdsam
   pip install -r requirements.txt
   git submodule update --init --recursive
   ```
3. Download 
    DINOv2(Vit-L) [checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)
    SAM(ViT-L) [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth).
     
    Place the donwdloaded weights in the  *weights* directory. If it does not exist, use command ``` mkdir weights ``` to create one.
   
## 3. Preparing Data
### 1. CrowdHuman

Download the CrowdHuman dataset from the [official website](https://www.crowdhuman.org/download.html). *Note that we only need the CrowdHuman_val.zip* and *annotation_val.odgt*. 
Extract and place the downdloaded zip files in the `dataset` directory and it should look like this:

```
crowdsam/
├── dataset/
│   └── crowdhuman/
│       ├── annotation_val.odgt
│       ├── Images
└── ...
```

Run the script to convert odgt file to json file.
```
python tools/crowdhuman2coco.py -o annotation_val.odgt -v -s val_visible.json -d dataset/crowdhuman
```
## 4. How to use

To start training the model, run the following command:
```bash
python train.py --config_file ./configs/config.yaml
```
Make sure to update the `config.yaml` file with the appropriate paths and parameters as needed.

To evaluate the model, use the following command:
```bash
python tools/batch_eval.py
```
This will run the evaluation script on the test dataset and output the results.

![demo1](figures/demo_2.jpg)
## Acknowlegement
We build our project based on the segment-anything and dinov2.

## Citation

You can cite our paper with such bibtex:
```bibtex
@article{cai2024crowd,
  title={Crowd-SAM: SAM as a Smart Annotator for Object Detection in Crowded Scenes},
  author={Cai, Zhi and Gao, Yingjie and Zheng, Yaoyan and Zhou, Nan and Huang, Di},
  journal={arXiv preprint arXiv:2407.11464},
  year={2024}
}
```

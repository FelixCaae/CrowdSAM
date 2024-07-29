
#  Crowd-SAM: SAM as a Smart Annotator for Object Detection in Crowded Scenes

Crowd-SAM is a novel few-shot object detection and segmentation method designed to handle crowded scenes. We combine SAM with the specifically designed efficient prompt  sampler and a mask selection PWD-Net to achieve fast and accurate pedestrian detection! Crowd-SAM achieves 78.4\% AP on the Crowd-Human benchmark with 10 supporting images which is comparable to supervised detectors. 

For more details, read the [paper here](https://arxiv.org/abs/2407.11464)

## Important notes
This repository is still under-working to be better for users. Feel free to ask any questions in the issue!

## Installation
To set up Crowd-SAM, follow these steps:
1. Create environment
```
conda create -n crowdsam python=3.9
```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crowd-sam.git
   cd crowd-sam
   pip install -r requirements.txt
   git submodule update
   ```
3. Download Pretrained DINOv2 weights and SAM weights:
   - Download DINOv2 [weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)
   - Download SAM [weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth).
     
   We use Vit-L for both models and please download the corresponding checkpoints.
   
## Preparing Data

Download the CrowdHuman dataset from the [official website](https://www.crowdhuman.org) and place it in the `data` directory:
```
crowd-sam/
  ├── datasets/
  │   └── crowdhuman/
  │       ├── annotation_train.odgt
  │       ├── annotation_val.odgt
  │       ├── Images
  └── ...
```

Run the script to convert odgt file to json file
```
python tools/crowdhuman2coco.py --odgt-path ./datasets/crowdhuman/annotation_train.odgt --visible --save_path ./datasets/crowdhuman/train_visible.json
python tools/crowdhuman2coco.py --odgt-path ./datasets/crowdhuman/annotation_val.odgt --visible --save_path ./datasets/crowdhuman/val_visible.json
python tools/crowdhuman2coco.py --odgt-path ./datasets/crowdhuman/annotation_val.odgt --visible --size 500 --save_path ./datasets/crowdhuman/midval_visible.json

```
## Training

To start training the model, run the following command:
```bash
python train.py --config_file ./configs/config.yaml
```
Make sure to update the `config.yaml` file with the appropriate paths and parameters as needed.

## Testing

To evaluate the model, use the following command:
```bash
python tools/batch_eval.py
```
This will run the evaluation script on the test dataset and output the results.


## Additional Information

For more details on the configuration options and usage, refer to the documentation provided in the `docs` directory. If you encounter any issues or have questions, please open an issue on the GitHub repository.

## Citation

If you use Crowd-SAM in your research, please cite our paper:
```bibtex
@inproceedings{crowdsam2024,
  title={Crowd-SAM: SAM as a Smart Annotator for Object Detection in Crowded Scenes},
  author={Zhi Cai, Yingjie Gao, Yaoyan Zheng, Nan Zhou, Di Huang},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024},
}
```

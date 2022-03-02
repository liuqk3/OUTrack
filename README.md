## Introduction

This is the official implementation of `Online Multi-Object Tracking with Unsupervised Re-Identification Learning and Occlusion Estimation`.
Note that:
* Because I am quite busy recently, the codes are prepared in a hurry, so they may look ugly.
* The work has been done in Seqtember 2019, so some hyper-parameters maybe not the same with those in the paper.

I will try to make the codes more accurate and beautiful when I am available.


## Installation
* Clone this repo, and we'll call the directory that you cloned as ${OUTrack}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
conda create -n OUTrack python=3.7
conda activate OUTrack
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${OUTrack}
pip install -r requirements.txt
```
For windows users, there maybe some errors. What you need to do is to comment `cython-box` in requirements.txt.
Then do as follows:
```
pip install -r requirements.txt
pip install numpy==1.19.3
somewhere=path/to/somewhere
cd $somewhere
git clone https://github.com/samson-wang/cython_bbox.git
cd cython_bbox
python setup.py build install
```

* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 
```
# git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```
* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

The CrowdHuman dataset is used for pretraining, it can be downloaded from their [official webpage](https://www.crowdhuman.org). After downloading, you should prepare the data in the following structure:
```
${OUTrack}/data/crowdhuman
   |——————images
   |        └——————train
   |        └——————val
   └------annotation_train.odgt
   └------annotation_val.odgt
```

We use MOTChallenge for training and evaluation. [MOT16](https://motchallenge.net/data/MOT16/), [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/) can be downloaded from the official webpage of MOT challenge. After downloading, you should prepare the data in the following structure:
```
${OUTrack}/data/MOT16
   |——————images
            └——————train
            └——————test

${OUTrack}/data/MOT17
   |——————images
            └——————train
            └——————test
${OUTrack}/data/MOT20
   |——————images
            └——————train
            └——————test
```
Then, you can run following command to get json style annotations:
```
cd src/tools
python gen_json_annotations.py
```

## Pretrained models and baseline model

Coming soon ...

## Training and Inference
* Download DLA-34 COCO pretranined CenterNet model weight from [here](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view), and put it to `OUTrack/models/ctdet_coco_dla_2x.pth`

* See [mot17.sh](experiments/mot17.sh) for example. Note the commands in different bash scripts are not exactly the same with those used in the paper.

Note that the `--debug` option is quite useful for debugging while training or visaulize the tracking results online! If you are in debug (i.e. `opt.debug > 0`) mode, you can press `Esc` to stop.

## Acknowledgement
A large part of the code is borrowed from [FairMOT](https://github.com/ifzhang/OUTrack) and [CenterTrack](https://github.com/xingyizhou/CenterTrack). Thanks for their wonderful works.

## Citation

```
@article{liu2022online,
  title={Online Multi-Object Tracking with Unsupervised Re-Identification Learning and Occlusion Estimation},
  author={Liu, Qiankun and Chen, Dongdong and Chu, Qi and Yuan, Lu and Liu, Bin and Zhang, Lei and Yu, Nenghai},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```


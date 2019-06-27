# YOLOv2_pytorch

A Python3.5/Pytorch implementation of YOLOv2:[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242). And the official implementation are available [here](http://pjreddie.com/yolo9000/).

### Prerequisites
* python 3.5.x
* pytorch 0.4.1
* tensorboardX
* pillow
* scipy
* numpy
* opencv3
* matplotlib
* easydict

### Installation
1. Clone this repository (Faster_RCNN_pytorch):
    
        git clone --recursive https://github.com/Jacqueline121/YOLOv2_pytorch.git

2. Install dependencies:
    
        cd YOLOv2_pytorch
        pip install -r requirements.txt

### Repo Organization
* config: define configuration information of Faster RCNN
* dataset: Scripts for creating, downloading, organizing datasets.
* yolo: Neural networks and components that form parts of YOLOv2.
* utils: tools package, containing some necessary functions.

### Train

#### Download PASCAL VOC data

1. Download the training, validation, test data:
    
        # download 2007 data
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

        # download 2012 data
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

2. Extract data into one directory named VOCdevkit
    
        # 2007 data
        tar xvf VOCtrainval_06-Nov-2007.tar
        tar xvf VOCtest_06-Nov-2007.tar
        tar xvf VOCdevkit_08-Jun-2007.tar

        # 2012 data
        tar xvf VOCtrainval_11-May-2012.tar

3. It should have this basic structure:
    
        $VOCdevkit/                           # development kit
        $VOCdevkit/VOCcode/                   # VOC utility code
        $VOCdevkit/VOC2007                    # image sets, annotations, etc.
        # ... and several other directories ...

4. Create symlinks for the PASCAL VOC dataset:
    
        cd YOLOv2_pytorch/dataset
        mkdir data
        cd data
        # 2007 data
        mkdir VOCdevkit2007
        cd VOCdevkit2007
        ln -s $VOCdevit/VOC2007 VOC2007

        # 2012 data
        mkdir VOCdevkit2012
        cd VOCdevkit2012
        ln -s $VOCdevit/VOC2012 VOC2012

#### Download pretrained ImageNet model
    cd YOLOv2_pytorch/yolo/
    mkdir pretrained
    cd pretrained
    # darknet19
    wget https://pjreddie.com/media/files/darknet19_448.weights


#### Train
    python train.py

### Test
    python test.py

If you want to visualize the detection result, you can use:
    
    python test.py --vis
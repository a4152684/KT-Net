## KT-Net: Knowledge Transfer for Unpaired 3D Shape Completion
- [Preprint paper](https://arxiv.org/abs/2111.11976).

## Requirements
- Ubuntu 14.04 or higher
- CUDA 10.0 or higher
- Python v3.7 or higher
- Pytorch v1.2 or higher

Specifically, The code has been tested with:
- Ubuntu 18.04, CUDA 10.2, python 3.8.15, Pytorch 1.6.0, GeForce RTX 2080Ti.

## Installation
- Create the conda environment.
  ```
  conda create -n kt-net python=3.8
  conda activate kt-net
  ```
Installation instructions for Ubuntu 18.04:
   * Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. Only this configurations has been tested:
     - Python 3.8.15, Pytorch 1.6.0
    * Follow <a href="https://pytorch.org/">Pytorch installation procedure</a>. Note that the version of cudatoolkit must be strictly consistent with the version of CUDA

- Intall some packages.
  ```
  pip install -r requirements
  ```
- Install EMD.
  ```
  cd net/util/emd_module
  python setup.py install
  cd ../../..
  ```
  
## Dataset & Pretrained model
- [3DEPN]();
- [CRN](https://github.com/junzhezhang/shape-inversion) Refer to ShapeInversion;
- [Real-World Data](https://github.com/xuelin-chen/pcl2pcl-gan-pub) Refer to Pcl2Pcl;
- [Pretrained Weights]().
Also, you can download them from [BaiduDisk](https://pan.baidu.com/s/13GoHmTJ-jqg1zBgRbIUmNQ)(Code:0di4). Please place the data to ```./dataset``` and the pretrained model to '''./pretrain'''.

## Train && Test
To train the model, you can edit the parameter in the file '''train_KT.sh''' and run the command:
  '''
  sh train_KT.sh
  '''

To test the model, you can edit the parameter in the file '''test_KT.sh''' and run the command:
  '''
  sh test_KT.sh
  '''
 
 ## Acknowledgement
The code is in part built on [MSC](https://github.com/ChrisWu1997/Multimodal-Shape-Completion). 
The original code of emd is rendered from [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion). 
The original code of chamfer3D is rendered from ["chamferDistancePytorch"](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/master/chamfer3D).

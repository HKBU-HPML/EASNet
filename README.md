# EASNet
EASNet: Searching Elastic and Accurate Network Architecture for Stereo Matching (ECCV 2022)
[[arXiv]]() [[Video]]()

PyTorch implementation of searching an efficient network architecture for stereo matching. Please cite the paper below if you use this project. Any suggestion, fork, and pull request is welcome. 
```BibTex
@inproceedings{
  wang2022easnet,
  title={EASNet: Searching Elastic and Accurate Network Architecture for Stereo Matching},
  author={Qiang Wang and Shaohuai Shi and Kaiyong Zhao and Xiaowen Chu},
  booktitle={European Conference on Computer Vision},
  year={2022},
  url={https://arxiv.org/abs/2207.09796}
}
```

## Setup

Python 3 dependencies:

* PyTorch 1.8+
* OpenMPI 4.0.1+
* matplotlib
* numpy
* imageio
* other necessary packages

We use the deformation module from AANet. Install the "deform_conv" package as follow.
```
cd ofa/stereo_matching/networks/deform_conv
sh build.sh
```

Our training scripts apply MPI to accelerate the training procedure. Please install OpenMPI 4.0.1 or above. 

## Searching EASNet
The main commands are summarized in "train.sh". One can use them accordingly. 

### Train the largest supernet.
```
# two nodes, four GPUs per node
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stereo.py \
           --task large
```
### Shrink the kernel size/depth/width/scale.
```
# two nodes, four GPUs per node
export TASK=kernel  # 'kernel', 'depth', 'width', 'scale'
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stereo.py \
           --task $TASK
```

## Finetuning on KITTI 2012/2015
```
# two nodes, four GPUs per node
export TASK=kitti12  # 'kitti12', 'kitti2015'
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stereo.py \
           --task $TASK
```

## Pretrained Models
| Scene Flow | KITTI 2012| KITTI 2015 |  
|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1c1iiADFmbi4hawC3JSa4TfcemxVd5hMU/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1lBI9Py82niQRTbmHk8XMWKcT0IOW583p/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1eQi0Cs4O6VG_XXWFsjZTyP6tgVXNcK7U/view?usp=sharing)|

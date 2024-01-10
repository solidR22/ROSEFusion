# RoseFusion_Comment
> RoseFusion算法注释

源仓库：https://github.com/jzhzhang/ROSEFusion

RoseFusion算法整体改自KinectFusion，只有**位姿估计**部分较之有较大改动，故仅注释**位姿估计**相关部分，如有纰漏，欢迎指正！
- 另：KinectFusion的详细注释请[参考](https://github.com/DreamWaterFound/KinectFusionAppLib_comments)@[Guoqing Liu](https://github.com/DreamWaterFound)


---

# ROSEFusion :rose:

This project is based on our SIGGRAPH 2021 paper, [ROSEFusion: **R**andom **O**ptimization for Online Den**SE** Reconstruction under Fast Camera Motion
](https://arxiv.org/abs/2105.05600).



## Introduction

ROSEFusion is proposed to tackle the difficulties in fast-motion camera tracking using random optimization with depth information only. Our method performs robust  camera tracking under fast camera motion at a real-time frame rate, without loop closure or global pose optimization.

 <p id="demo1" align="center"> 
  <img src="assets/intro.gif" />
 </p>

## Installation 

### tested environment 1 
Our code is based on C++ and CUDA with the support of:
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (tested on v0.6)
- OpenCV with CUDA (v.4.5 is required, for instance you can follow the [link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7))  
- Eigen (tested on 3.3.9)
- CUDA (tested with V11.1, 11.4)


The code has been tested with Nvidia GeForce RTX 2080 SUPER on Ubuntu 16.04. 

### tested environment 2
Our code is based on C++ and CUDA with the support of:
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (tested on v0.6)
- OpenCV with CUDA (v.4.5.5 is required, for instance you can follow the [link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7))  
- Eigen (tested on 3.4.0)
- CUDA (tested with V11.8)

The code has been tested with Nvidia GeForce RTX 3060 SUPER on Ubuntu 20.04. 

**Please make sure the architecture ```(sm_xx and compute_xx)``` in the [L22 of CMakeLists.txt](CMakeLists.txt#L22) is compatible with your own graphics card.**

## [Option] Test with Docker

We have already upload a docker image with all the lib, code and data. You can download the docker image from the [One drive](https://1drv.ms/u/s!AvuKnc9E9hmqhh_D_8pksC-cC2ma?e=K5DFnN).

### **Prepare**
Please make sure you have successfully installed the [docker](https://www.docker.com/) and [nvidia docker](github.com/NVIDIA/nvidia-docker). and once the environment is ready, you can use following commands to boot the docker image:
```
sudo docker load -i rosefusion_docker.tar 
sudo docker run -it  --gpus all jiazhao/rosefusion:v7 /bin/bash
```


And please check the architecture in the L22 of ``` /home/code/ROSEFusion-main/CMakeList.txt``` is compatible with your own graphics card. If not, change the sm_xx and compute_xx, then rebuild the ROSEFusion.


### **QuickStart**
We have already configured the path and data in the docker image. You can simply run "run_example.sh" and "run_stairwell.sh" at  ```/home/code/ROSEFusion-main/build``` and the trajectory and reconstuciton would be saved in ```/home/code/rosefusion_xxx_data```. 



## Configuration File
We use the following configuration files to make the parameters setting easier. There are four types of configuration files.

- **seq_generation_config.yaml:** data information 
- **camera_config.yaml:** camera and image information.
- **data_config.yaml:** output path, sequence file path and parameters of the volume.
- **controller_config.yaml:** visualization, results saving and parameters of tracking.

The **seq_generation_config.yaml** is only used for data preparation, and the other three types of configuration files are necessary to run the ROSEFusion. We have alreay prepared some configuration files of some common datasets, you can check the details in `[type]_config/` directory. You can change the parameters to fit your own dataset.

## Data Preparation
The details of data preparation can be found in [src/seq_gen.cpp](src/seq_gen.cpp). By using the *seq_generation_config.yaml* introduced above, you can run the script as:
```
./seq_gen  sequence_information.yaml
```
Once finished, there would be a `.seq` file which could be used for future reconstruction.


## Particle Swarm Template
We share the same pre-sampled PST as our paper. Each PST is saved as an N×6 image and the N means the number of particles. You can find the ``.tiff`` images in [PST dicrectory](/PST), and please change the PST path in ``controller_config.yaml `` with your own path.

## Running
Finally, to run the ROSEFusion, you need to provide the `camera_config.yaml`, `data_config.yaml` and `controller_config.yaml`. We already share configuration files of many common datasets in `./camera_config`, `./data_config`, `/controller_config`. All the parameters of configuration files can be modified as you want. Once you have all the required files, you can run the ROSEFsuion as:
```
./ROSEFsuion  your_camera_config.yaml your_data_config.yaml your_controller_config.yaml
```
For a quick start, you can download and use a small size synthesis [seq file with related configuration files](https://1drv.ms/u/s!AvuKnc9E9hmqhhhkR1_FWUlDNxfO?e=M8S4iI). Here is a preview.


 <p id="demo1" align="center"> 
  <img src="assets/example.gif" />
 </p>

## FastCaMo Dataset
We present the **Fast** **Ca**mera **Mo**tion dataset, which contains both synthetic and real captured sequences. For more details, please refer to the paper.
### FastCaMo-Synth
With 10 diverse room-scale scenes from [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset), we render the color images and depth maps along the synthetic trajectories. The raw sequences are provided in [FastCaMo-synth-data(raw).zip](https://1drv.ms/u/s!AvuKnc9E9hmqhgljQdXKJECStZ-W?e=wWD2Tz), and we also provide the [FastCaMo-synth-data(noise).zip](https://1drv.ms/u/s!AvuKnc9E9hmqhgoktvWzmX_x6v2k?e=rO3MWv) with synthetic noise and motion blur. We use the same noise model as [simkinect](https://github.com/ankurhanda/simkinect). For evaluation, you can download the ground truth [trajectories](https://1drv.ms/u/s!AvuKnc9E9hmqhghwEOxFW4Za4orv?e=q9HRSK).

### FastCaMo-Real
It contains 12 [real captured RGB-D sequences](https://1drv.ms/u/s!AvuKnc9E9hmqhXtEpQ1fMViDRh6x?e=3sCYft) under fast camera motions. Each sequence is recorded in a challenging scene like gym or stairwell by using [Azure Kinect DK](https://azure.microsoft.com/en-us/services/kinect-dk/). We provide accurate dense reconstructions as ground truth, which are modeled with the high-end laser scanner. However, the original models are extremely large, and we utilized the built-in spatial downsample algorithm from cloudcompare. You can download the sub-sampled [models of FastCaMo-real form here](https://1drv.ms/u/s!AvuKnc9E9hmqhgtSGIIH1FL5V2b1?e=HhNB0c). 

 <p id="demo1" align="center"> 
  <img src="assets/fastcamo-real.gif" />
 </p>

## Citation
If you find our work useful in your research, please consider citing:
```
@article {zhang_sig21,
    title = {ROSEFusion: Random Optimization for Online Dense Reconstruction under Fast Camera Motion},
    author = {Jiazhao Zhang and Chenyang Zhu and Lintao Zheng and Kai Xu},
    journal = {ACM Transactions on Graphics (SIGGRAPH 2021)},
    volume = {40},
    number = {4},
    year = {2021}
}
```

## Rum TUM datasets
```shell
./runTum ../camera_config/TUM2.yaml ../dataset_config/TUM\ RGBD/fr2_desk.yaml ../controller_config/controller.yaml
```

## FAQ
Q: The ```Frame could not be processed``` error is reported when running the example data.

A: Please make sure you have correctly installed the environment:
- Check the ```ompute_xx,code=sm_xx``` and make sure it is valid for your GPU device. 
- Build the opencv with ```-D WITH_CUDA=ON``` and make sure the ```-D CUDA_ARCH_BIN=x.x``` is valid for your GPU device.


## Acknowledgments
Our code is inspired by [KinectFusionLib](https://github.com/chrdiller/KinectFusionLib).

This is an open-source version of ROSEFusion, some functions have been rewritten to avoid certain license. It would not be expected to reproduce the result exactly, but the result is almost the same.
## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.

## Contact
If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.


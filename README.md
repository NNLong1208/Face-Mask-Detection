#### Table of contents
1. [Introduction](#introduction)
2. [Details](#details)
   - [Dataset](#Dataset)
   - [Proposedmodel](#Proposed_model)
   - [Mobilenet](#Mobilenet)
   - [YOLOV5](#YOLOV5)
   - [Face Detection](#Face_Detection)
   - [ Pose Estimation](#Pose_Estimation)
3. [QuickStart](#QuickStart)
4. [Results](#Results)
5. [Contact](#Contact)
<p align="center">
  <h1 align="center", id="introduction">NEU-Mask</h1>
</p>

- We researched and created a technique to detect mask wear in this project. We chose [`Mobilenet`](#Mobilenet) after examining a variety of options. Other models, such as [`YOLOv5`](#YOLOV5), [`Face Detection`](#Face_Detection) and [`Pose Estimation`](#Face_Detection), have also been merged. NEU-MASK is created by combining four models.

- Our project is capable of operating in real time. The processing rate is up to 12 FPS. We also provide a 16,000-image [`dataset`](#dataset) separated into two classes: masks and no masks.

## Details <a name="Details"></a>

### Dataset <a name="Dataset"></a>
- The dataset used to train the model is a compilation of images from various sources. The majority of the data set is gathered on major social networking sites such as Facebook and Twitter and significant search engines like Google, Bing, Baidu, and so on. Furthermore, a huge number of pictures from garment factory No.10 (Hanoi, Vietnam) featuring images of workers wearing masks shot from various perspectives have been collected. In addition, in order to improve the variety and accuracy of the prediction model, the dataset gathers photos from different countries, ages, genders, and shooting angles. To reduce the possibility of bias, we looked for and collected additional images of persons who were not wearing masks but were covering their lips with their hands or had a beard to include in the collection of images without masks. You can download dataset at [here](https://drive.google.com/file/d/1-pWX25WnebaSvqCNuVc4zvR5mY6fsUES/view?usp=sharing)

### Proposed model  <a name="Proposed_model"></a>
- To build NEU-Mask we use 4 models which are [`Mobilenet`](#Mobilenet), [`YOLOv5`](#YOLOV5), [`Face Detection`](#Face_Detection) and [`Pose Estimation`](#Pose_Estimation). NEU-Mask is divided into 2 stage . The first stage is combined by [`Face Detection`](#Face_Detection) and [`Mobilenet`](#Mobilenet). Stage 2 is combined by [`Yolov5`](#YOLOv5) and [`Pose Estimation`](#Pose_Estimation).  The wearing of a fake mask will be scrutinized in step 2 after passing phase 1. Download all model at [here](https://drive.google.com/file/d/10Jm4ztCeV9dqVMVGzLP9B3iOyUb2EKLJ/view?usp=sharing)
![](https://github.com/NNLong1208/Face-Mask-Detection/blob/master/Img/NEUMASK.png)
### Mobilenet <a name="Mobilenet"></a>
- We used MobileNet as a base model for NEU-MASK, then vectorized the output of the MobileNet model to integrate the Dense layer into the neural network. Afterward, we utilized another Dense layer with `512 neurons` with `Sigmoid activation`. Finally, we used the Dense layer, which has `one neuron` with a `softmax activation` function, to classify whether the input image has a mask. To aid the model in preventing overfitting, we utilized a Dropout layer in the midst of the two Dense layers with a `p = 0.3% ratio`.
![](https://github.com/NNLong1208/Face-Mask-Detection/blob/master/Img/Mobilenet.png)
- To make Mobilenet usage decisions, we experiment with a variety of models. The specific outcomes are listed in the table below:

|         Model       |   Accuracy   |     FPS     |         Model       |   Accuracy   |     FPS     |         Model       |   Accuracy   |     FPS     | 
|---------------------|:------------:|:-----------:|---------------------|:------------:|:-----------:|---------------------|:------------:|:-----------:|
|       Xception      |    99.41%    |     2.7     |      ResNet101V2    |    99.67%    |     1.8     |      DenseNet169    |    99.00%    |     4.1     |
|       ResNet50      |    99.47%    |     2.1     |      InceptionV3    |    99.35%    |     2.2     |      DenseNet201    |    99.26%    |     3.0     |
|       ResNet101     |    99.47%    |     1.8     |      MobileNetV2    |    95,82%    |     8.6     |      NASNetLarge    |    99.41%    |     1.1     |
|       ResNet152     |    99.76%    |     1.4     |     `MobileNet`     |   `99.41%`   |     8.2     |      NASNetMobile   |    98.39%    |     7.7     |
|       ResNet50V2    |    99.73%    |     2.1     |      DenseNet121    |    99.05%    |     5.2     |   InceptionResNetV2 |    99.58%    |     1.6     |
- After considering the processing speed and accuracy of the model, we have chosen Mobilenet because of its accuracy and suitable processing speed. Next we converted from keras to Openvino to increase the processing speed from `8.2 FPS` to `32 FPS `

### YOLOV5 <a name="YOLOV5"></a>
- YOLOv5 is a collection of object detection architectures and models pre-trained on the COCO dataset. It includes: YOLOv5s, YOLOv5m, yolov5l, YOLOv5x, ... We use yolov5m for the purpose of detecting the location of the masks in the image. More information about the model you can see [here](https://github.com/ultralytics/yolov5) 
- We train yolov5 with class with mask of the dataset. After that, yOLOv5 has a `mAP` of `0.81` when training on this dataset. 

### Face Detection <a name="Face_Detection"></a>
- We use this model built and optimized by Intel. Face detector based on SqueezeNet light (half-channels) as a backbone with a single SSD for indoor/outdoor scenes shot by a front-facing camera. The backbone consists of fire modules to reduce the number of computations. The single SSD head from 1/16 scale feature map has nine clustered prior boxes. More information about the model you can see [here](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0004)  

- In addition, we have used the landmarks regression model. It is also built by Intel. It has a classic convolutional design: stacked 3x3 convolutions, batch normalizations, PReLU activations, and poolings. Final regression is done by the global depthwise pooling head and FullyConnected layers. The model predicts five facial landmarks: two eyes, nose, and two lip corners. More information about the model you can see [here](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/landmarks-regression-retail-0009) 

### Pose Estimation <a name="Pose_Estimation"></a>
- Pose Estimation identifies human poses for each person in the image by detecting a skeleton (which comprises of keypoints and relationships between them). Ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles are among the `18 keypoints` that may be found in the position. This algorithm achieves `40% AP` for the single scale inference (no flip or post processing) on the COCO 2017 Keypoint Detection validation set. More information about the model you can see [here](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) 

## QuickStart <a name="QuickStart"></a>
You can use git:
```
git clone https://github.com/NNLong1208/Face-Mask-Detection
cd Face-Mask-Detection
pip3 install -r requirements.txt
python main.py --camera <0 or link of image>
```
or pypi : update later

## Results <a name="Results"></a>
- We tested with many different cases: 
                                    
![](https://github.com/NNLong1208/Face-Mask-Detection/blob/master/Img/res.gif)
## Contact <a name="Contact"></a>
- Supervisor: [Tuan Nguyen](https://www.facebook.com/nttuan8)
- Team Members: [Long Nguyen](https://www.facebook.com/profile.php?id=100008475522373), [Tung Nguyen](https://www.facebook.com/gnutn0s), [Thao Nguyen](), [Trang Tran](https://www.facebook.com/concanoc1), [Phuong Duong]() 

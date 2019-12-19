# CS_IOC5008_0856078_HW4
## Brief Introduction
　The task of this homework is to do instance segmentation on Tiny PASCAL VOC dataset. There are 1349 training images and 100 test images in this dataset with 20 common object classes. I use detectron2 to train Mask R-CNN with a ResNext101 + FPN (feature pyramid networks) backbone. The result of mask mAP (IoU = 0.5) is 0.5229.
## Files Description
* **demo.py:** This .py file is based on the Colab Notebook of Detecton2. It only contains the code of predicted part. After building, you can run this file to check whether Detectron2 can work or not.  
* **HW3_train.py:** It contains training part and validation part. You can tune some hyperparameters here. Amd make sure the file path is modified correctly by you.  
* **HW3_test.py:** It contains testing part.
## Hardware and OS
* Windows10
* 1x NVIDIA 1080ti
## Reproducing Submission
#### File path structure
```
Detrctron2
  +- train_images
  |  +- images
  |  +- annotation
  +- test_images
  |  +- images
  |  +- annotation
  +- output
  |  +- my pretrained model
  +- ImageNet pretrained model
  +- demo.py
  +- HW4_train.py
  +- HW4_test.py
  +- submission files
```
To reproduce my submission, do the following steps:   
#### 1. Install detectron2
This repository do not contain Detectron2. You should build Detectron2 first. If your OS is linux, follow [Detectron2](https://github.com/facebookresearch/detectron2) directly. If your OS is win10, you can follow the [Detectron2 on windows](https://github.com/conansherry/detectron2). And you also need to install full version of CUDA instead of cudatoolkit. After building Detectron2, place the files of this repository into Detectron2 main folder.
#### 2. Prepare data
Download the dataset from this [Google Drive](https://drive.google.com/drive/u/0/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK). Create two folders 'train_images' and 'test_images' in the main folder. Put the trainging and testing data (include images and annotations) into these two folders respectively.
#### 3. Download pretrained model
If you want to retrain the model, download the ImageNet pretrained model from this [link](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl).(from Detectron2 model zoo)  
If you only want to reproduce my submission without retraining, download my pretrained model from this [Google Drive](https://drive.google.com/open?id=174ARrf4Oj7SuQ7nOdXcxzuXDYy8Nso-R).
#### 4. Inference
Run train.py for training.  
Run test.py for testing.
## Reference from GitHub
[Detectron2](https://github.com/facebookresearch/detectron2)  
[Detectron2 on windows](https://github.com/conansherry/detectron2)
## Findings or Summary
　At first, I think if I don’t use any technique to address overfitting, the result would be at most about baseline. However, mAP actually achieves about 0.5. I even use COCO pretrained model to validate the ImageNet pretrained model I utilize. The result of COCO is about 0.7, which much higher than ImageNet and only need 50000 iterations. Thus, I think the ImageNet pretrained model is correct and overfitting does not exist in this homework, even if there are only 1349 training images. I also upload the test submission with different iterations. The results of more iteration usually have better validation and testing performance, which also means overfitting does not happen. I believe this is because ImageNet pretrained model is powerful enough. We can see ImageNet pretrained model as a feature extractor. And it occupies almost the entire network. The mask branch only have few layers. Besides, ImageNet have 1000 classes (include 20 classes in Tiny PASCAL VOC) and a huge number of data. Therefore, I think 1349 images are enough to train mask branch because we don’t need to retrain the feature extractor. The other reason I believe is similarity between training set and testing set. These two reasons are just assumptions. I do not prove these yet.

# CS_IOC5008_0856078_HW4
## Files Description
* **demo.py:** This .py file is based on the Colab Notebook of Detecton2. It only contains the code of predicted part. After building, you can run this file to check whether Detectron2 can work or not.  
* **HW3_train.py:** It contains training part and validation part. You can tune some hyperparameters here. Amd make sure the file path is modified correctly by you.  
* **HW3_test.py:** It contains testing part.
## Hardware and OS
* Windows10
* 1x NVIDIA 1080ti
## Reproducing Submission
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

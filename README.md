# keras_emotion_classify
a keras implement of emotion classifier.


## Test environment
* python 3.7.3
* keras 2.2.4
* OpenCV 4.0.0
* tensorflow 1.12.0
* cuda 10.0.130
* cudnn 7.5.1


## How to run?
1. Download dataset fer2013.tar.gz.
2. process dataset.
```sh
python prepare_data.py
```
3. train your model.
```sh
python emotion_train.py --dataset data/fer2013 --checkpoint ckpt/CapsuleNet -b 128 --network CapsuleNet
```


## Dataset summary
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/fer2013_summary.png)


## Test summary
all network run 70 epochs.
|Network		|optimizer	|learning rate	|acc	|val_acc	|
|---------------|-----------|---------------|-------|-----------|
|VGG16			|adam		|0.0001			|89%	|64%		|
|VGG19			|adam		|0.0001			|89%	|64%		|
|ResNet50		|adam		|0.0001			|75%	|50%		|
|DenseNet121	|adam		|0.0001			|91%	|55%		|
|DenseNet201	|adam		|0.0001			|92%	|55%		|
|MobileNetV2	|adam		|0.0001			|65%	|47%		|
|CapsuleNet		|adam		|0.0001			|95%	|66%		|
|CapsuleResNet	|adam		|0.0001			|88%	|63%			|
|DenseNet201	|adam		|0.002			|88%	|63%		|
|CapsuleResNet	|adam		|0.0005			|86%	|64%		|
|VGG16			|sgd		|0.01			|93%	|62%		|

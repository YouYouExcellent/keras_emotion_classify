# keras_emotion_classify
a keras implement of emotion classifier.

# 联系作者
* 邮件: youyou630@163.com
* QQ: 603722900

## Test environment
* python 3.7.3
* keras 2.2.4
* OpenCV 4.0.0
* tensorflow 1.12.0
* cuda 10.0.130
* cudnn 7.5.1

## DATASETs

### FER2013 dataset

#### How to run?
1. Download dataset [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. process dataset.
```sh
python prepare_data.py
```
3. train your model.
```sh
python emotion_train.py --dataset data/fer2013 --checkpoint ckpt/CapsuleNet -b 128 --network CapsuleNet
```


#### Dataset summary
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/fer2013_summary.png)


#### Results

all networks run 70 epochs.

| Network		| optimizer	| learning rate	| acc	| val_acc	|
| --------- | --------- | ------------- | --- | ------- |
| VGG16			| adam		| 0.0001			| 89%	| 64%		|
| VGG19			| adam		| 0.0001			| 89%	| 64%		|
| ResNet50		| adam		| 0.0001			| 75%	| 50%		|
| DenseNet121	| adam		| 0.0001			| 91%	| 55%		|
| DenseNet201	| adam		| 0.0001			| 92%	| 55%		|
| MobileNetV2	| adam		| 0.0001			| 65%	| 47%		|
| CapsuleNet		| adam		| 0.0001			| 95%	| 66%		|
| CapsuleResNet	| adam		| 0.0001			| 88%	| 63%			|
| DenseNet201	| adam		| 0.002			| 88%	| 63%		|
| CapsuleNet		| adam		| 0.0005			| 92%	| 65%		|
| CapsuleResNet	| adam		| 0.0005			| 86%	| 64%		|
| VGG16			| sgd		| 0.01			|93%	| 62%		|

### FERPLUS dataset

#### How to run?
1. Download dataset FERPlus.
```sh
git clone https://github.com/microsoft/FERPlus.git
```
2. process dataset.
```sh
cd FERPlus/src
python generate_training_data.py -d /dst/dataset/path/ferplus -fer you/fer2013/dataset/path/fer2013.csv -ferplus ../fer2013new.csv
cd you/own/path/keras_emotion_classify
python prepare_ferplus_data.py --image /dst/dataset/path/ferplus --label FERPlus/data --dst final/dst/path

```
3. train your model.
```sh
python emotion_train.py --dataset final/dst/path --checkpoint ckpt/CapsuleNet -b 128 --network CapsuleNet
```


#### Dataset summary
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_summary.png)


#### Results

all networks run 70 epochs.

| Network		| optimizer	| learning rate	| acc	| val_acc	|
| --------- | --------- | ------------- | --- | ------- |
| CapsuleNet	| adam		| 0.0001		| 96%	| 82%		|
| VGG16	| adam		| 0.0001		| 95%	| 81%		|

#### mAP
use CapsuleNet model

![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_angry.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_disgust.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_fear.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_happy.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_sad.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_surprise.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_neutral.png)
![](https://github.com/YouYouExcellent/keras_emotion_classify/blob/master/ferplus_ap_contempt.png)

| emotion	| angry | disgust| fear	| happy	| sad | surprise | neutral | contempt |
| --------- | --------- | ------------- | --- | ------- |--- |--- |--- |--- |
| AP | 65% | 0.97%		| 26.4%	| 89.9%		| 25.4% | 77.8% | 74.9% | 26.3% |


mAP = 48.3%


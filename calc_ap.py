import argparse
import os
import common
import glob
import numpy as np
from matplotlib import pyplot as plt
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
from keras.models import load_model
from capsule_net import Capsule
from keras import backend as K

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']

networks = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'CapsuleResNet',
            'MobileNet', 'MobileNetV2', 'CapsuleNet', 'ResNet50',
            'VGG16', 'VGG19']

def capsule_loss(y_true, y_pred):
    return y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2

class CALCAP(common.COMMON):
    model = None
    def __init__(self, args):
        self.dataset = args.dataset
        self.checkpoint = args.checkpoint
        self.batch_size = args.batch_size
        self.emotion = args.emotion
        self.suffix = args.image_suffix
        self.image_size = args.image_size
        self.network = args.network
        self.show = args.show
        if not os.path.exists(self.checkpoint):
            self.log_error('Checkpoint path {} is not exist'.format(self.checkpoint))

    def load_model(self):
        if self.network not in ['CapsuleNet', 'CapsuleResNet']:
            self.model = load_model(os.path.join(self.checkpoint, 'model.h5'))
        else:
            self.model = load_model(os.path.join(self.checkpoint, 'model.h5'),
                                    custom_objects={'Capsule': Capsule,
                                                    'capsule_loss': capsule_loss})

    def start(self):
        final = np.array([])
        y_true = np.array([])
        self.load_model()
        dirs = os.listdir(self.dataset)
        idx = emotions.index(self.emotion)
        count = 0
        for d in dirs:
            imgs = glob.glob(os.path.join(self.dataset, d, '*.{}'.format(self.suffix)))
            count += len(imgs)
            one_true = np.eye(len(dirs))[int(d)]
            while imgs:
                images = []
                batch = imgs[:self.batch_size]
                imgs = imgs[self.batch_size:]
                for i in batch:
                    img = image.load_img(i, color_mode='grayscale', target_size=(self.image_size, self.image_size))
                    x = image.img_to_array(img)
                    images.append(x)
                x = np.array(images)
                y = self.model.predict(x)
                final = y if final.size == 0 else np.append(final, y, axis=0)
                bt = np.tile(one_true, (len(batch),1))
                y_true = bt if y_true.size == 0 else np.append(y_true, bt, axis=0)
        final = np.append(final, y_true[:,idx].reshape(count, 1), axis=1)
        aidx = list(np.argsort(final[:,idx]))
        aidx.reverse()
        final = final[aidx]
        gt = np.where(final[:,-1]==1.)[0]
        x_recall = []
        y_precision = []
        total = gt.size
        for i,e in enumerate(gt):
            x_recall.append((i+1)/float(total))
            y_precision.append((i+1)/float(e+1))
        mAP = np.mean(y_precision)
        if self.show:
            plt.title('{} AP | mAP ({:.3})'.format(self.emotion, mAP))
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.plot(x_recall, y_precision)
            plt.show()
        else:
            print('emotion: {} mAP: {:.3}'.format(self.emotion, mAP))

def get_args():
    parser = argparse.ArgumentParser(description="Calculate AP.")
    parser.add_argument('--dataset', '-d',
                        help="path to dataset.")
    parser.add_argument('--checkpoint', '-c',
                        help="checkpoint directory.")
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help="batch size")
    parser.add_argument('--image-suffix', '-i', default='png',
                        help="image type")
    parser.add_argument('--image-size', '-s', type=int, default=48,
                        help="image width or height")
    parser.add_argument('--emotion', '-e', default='happy', choices=emotions,
                        help="emotion")
    parser.add_argument('--network', '-n', default='CapsuleNet', choices=networks,
                        help="network: {}".format(', '.join(networks)))
    parser.add_argument('--show', action='store_true', default=False,
                        help="show figure")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    e = CALCAP(args)
    e.start()

if __name__ == '__main__':
    main()

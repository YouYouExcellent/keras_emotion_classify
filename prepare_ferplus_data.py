import os
import csv
import shutil
import argparse
from common import COMMON

fer2013_em = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ferplus_em = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']

class FERPLUS(COMMON):
    fer_dirs = {'train':'FER2013Train', 'test':'FER2013Test', 'val':'FER2013Valid'}
    def __init__(self, args):
        self.image = args.image
        self.label = args.label
        self.dst   = args.dst

    def path_valid(self):
        if not os.path.exists(self.image):
            self.log_error('FERPLUS image path {} not found!'.format(self.image))

        ps = [os.path.join(self.image, i) for i in self.fer_dirs.values()]
        for p in ps:
            if not os.path.exists(p):
                self.log_error('FERPLUS image path {} not found!'.format(p))

        if not os.path.exists(self.label):
            self.log_error('FERPLUS label path {} not found!'.format(self.label))

        ps = [os.path.join(self.label, i, 'label.csv') for i in self.fer_dirs.values()]
        for p in ps:
            if not os.path.exists(p):
                self.log_error('FERPLUS label path {} not found!'.format(p))

        if not os.path.exists(self.dst):
            os.makedirs(self.dst)
        ps = [os.path.join(self.dst, i) for i in self.fer_dirs]
        for p in ps:
            if not os.path.exists(p):
                os.makedirs(p)
            for i in range(len(ferplus_em)):
                if not os.path.exists(os.path.join(p, str(i))):
                    os.makedirs(os.path.join(p, str(i)))

    def copy_image(self):
        for od, pd in self.fer_dirs.items():
            src = os.path.join(self.image, pd)
            dst = os.path.join(self.dst, od)
            with open(os.path.join(self.label, pd, 'label.csv')) as f:
                lines = csv.reader(f)
                for line in lines:
                    img,_,n,h,s,sad,a,ds,fear,c,_,_=line
                    if img:
                        arr = [n,h,s,sad,a,ds,fear,c]
                        m = max(arr)
                        idx = arr.index(m)
                        em = ferplus_em[idx]
                        i = fer2013_em.index(em) if em != 'contempt' else 7
                        shutil.copyfile(os.path.join(src, img), os.path.join(dst, str(i), img))

    def start(self):
        self.path_valid()
        self.copy_image()


def main():
    parser = argparse.ArgumentParser(description='process ferplus dataset.')
    parser.add_argument('--image',
                        help="ferplus image path.")
    parser.add_argument('--label',
                        help="ferplus lable path.")
    parser.add_argument('--dst',
                        help="dst path")
    args = parser.parse_args()
    
    ferplus = FERPLUS(args)
    ferplus.start()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', '-p',
                    help="dataset path.")
parser.add_argument('--dataset', '-d', default='train',
                    help="dataset type.")
args = parser.parse_args()

if os.path.exists(args.dataset_path):
    dirs = os.listdir(os.path.join(args.dataset_path, args.dataset))
    name_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    if len(dirs) == 8:
        name_list.append('contempt')
    num_list = []
    for i in range(len(name_list)):
        num_list.append(len(os.listdir(os.path.join(args.dataset_path, args.dataset, str(i)))))

    rects=plt.bar(range(len(num_list)), num_list)

    index=[0,1,2,3]
    index=[float(c)+0.4 for c in index]
    plt.xticks(range(len(name_list)), name_list)
    plt.ylabel("count")
    plt.show()

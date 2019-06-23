import matplotlib.pyplot as plt
import os
 
name_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_list = []
for i in range(len(name_list)):
    num_list.append(len(os.listdir(os.path.join('data', 'fer2013', 'train', str(i)))))

rects=plt.bar(range(len(num_list)), num_list)

index=[0,1,2,3]
index=[float(c)+0.4 for c in index]
plt.xticks(range(len(name_list)), name_list)
plt.ylabel("count")
plt.show()

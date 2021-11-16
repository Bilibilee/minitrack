import numpy as np
import random
total_filepath=r'.\helmet_total_anno.txt'
train_filepath=r'.\helmet_train_anno.txt'
val_filepath=r'.\helmet_val_anno.txt'
test_filepath=r'.\helmet_test_anno.txt'


with open(total_filepath) as f:
    lines=f.readlines()

np.random.seed(1)

#每隔四帧取一张图片
newlines=[]
for i in range(len(lines)):
    if i % 4==0:
        newlines.append(lines[i])
lines=newlines
np.random.shuffle(lines)#用于随机打乱，原地修改

trainval_percent=0.9#拿出trainval_percent做训练集和验证集,剩下的做测试集
train_percent=0.9#train_percent表示训练集占训练集和测试集的比例

num=len(lines)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

print("train and val size", tv)
print("traub suze", tr)

ftest = open(test_filepath, 'w')
ftrain = open(train_filepath, 'w')
fval = open(val_filepath, 'w')

for i in list:
    anno = lines[i]
    #anno=anno.replace('D:\\Dataset\\','C:\\Users\\xdtech\\Desktop\\')
    print(anno)
    if i in trainval:
        if i in train:
            ftrain.write(anno)
        else:
            fval.write(anno)
    else:
        ftest.write(anno)

ftrain.close()
fval.close()
ftest.close()
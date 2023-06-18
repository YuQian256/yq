import mindspore
import numpy as np
from mindspore import nn
from mindspore.dataset import vision,transforms
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt

# ## 数据集加载
# 我们使用**Mnist**数据集作为样例，介绍使用`mindspore.dataset`进行加载的方法。
# `mindspore.dataset`提供的接口**仅支持解压后的数据文件**，因此我们使用`download`库下载数据集并解压。

# In[2]:
# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

# 压缩文件删除后，直接加载，可以看到其数据类型为MnistDataset。
# In[3]:
train_dataset = MnistDataset("MNIST_Data/train", shuffle=False)
print(type(train_dataset))

#数据下载完成后，获得数据集对象
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
#打印数据集中的数据列名，用于dataset的预处理
print(train_dataset.get_col_names())

#MindSpore的dataset使用数据流水线(Data Processing Pipeline),需指定map,btach,shuffle等操作。这里使用map对图像数据及标签进行变换处理，
#然后将处理好的数据集打包为大小为64的batch
def datapipe(dataset,batch_size):
      image_transforms = [
            vision.Rescale(1.0/255.0,0),
            vision.Normalize(mean=(0.1307,),std=(0.3081,)),
            vision.HWC2CHW()
      ]
      label_transform = transforms.TypeCast(mindspore.int32)

      dataset = dataset.map(image_transforms,'image')
      dataset = dataset.map(label_transform,'label')
      dataset = dataset.batch(batch_size)
      return dataset
train_dataset = datapipe(train_dataset,64)
test_dataset = datapipe(test_dataset,64)

#使用create_tuple_iterator或create——dict_iterator对数据集进行迭代
for image,label in test_dataset.create_tuple_iterator():
      print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
      print(f"Shape of label: {label.shape} {label.dtype}")
      break


for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break

#网络构建 mindspore.nn类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，可以继承nn.Cell类，并重写_init_方法和construct方法。
# _init_包含所有网络层的定义，construct中包含数据(Tensor)的变换过程(即计算图的构造过程)
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)







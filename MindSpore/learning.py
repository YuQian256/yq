import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor,ops
from mindspore.common.initializer import One,Normal

#根据数据创建张量，数据类型可以设置或者通过框架自动推断
data = [1,0,1,0]
x_data = Tensor(data)
print(x_data)

#从NumPy数组创建张量
np_array = np.array(data)
x_np = Tensor(np_array)

#使用init初始化器构造张量
#当使用init初始化器对张量进行初始化时，传入的参数有init、shape、dtype
#init：支持传入initializer的子类  shape:支持传入list、tuple、int  dtype：支持传入mindspore.dtype
tensor1 = mindspore.Tensor(shape=(2,2),dtype=mindspore.float32,init=One())
tensor2 = mindspore.Tensor(shape=(2,2),dtype=mindspore.float32,init=Normal())
print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)

#继承另一个张量的属性，形成新的张量
x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")

#张量索引：Tensor索引与Numpy索引类似，索引从0开始编制，负索引表示按倒序编制，冒号：和...用于对数据的切片
tensor = Tensor(np.array([[0,1],[2,3]]).astype(np.float32))
print("First row: {}".format(tensor[0]))   #First row: [0. 1.]
print("value of bottom right corner: {}".format(tensor[1, 1]))  #value of bottom right corner: 3.0
print("Last column: {}".format(tensor[:, -1])) #Last column: [1. 3.]
print("First column: {}".format(tensor[..., 0]))  #First column: [0. 2.]
print("First row:{}".format(tensor[0,:]))  #First row:[0. 1.]
print("Last row:{}".format(tensor[-1,:]))  #Last row:[2. 3.]

#张量运算
x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add) #add: [5. 7. 9.]
print("sub:", output_sub) #sub: [-3. -3. -3.]
print("mul:", output_mul) #mul: [ 4. 10. 18.]
print("div:", output_div) #div: [4.  2.5 2. ]
print("mod:", output_mod) #mod: [0. 1. 0.]
print("floordiv:", output_floordiv) #整数除法  floordiv: [4. 2. 2.]

#Concat将给定维度上的一系列张量连接起来
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

print(output)
print("shape:\n", output.shape)
#[[0. 1.]   当axis=0时是列
# [2. 3.]
# [4. 5.]
# [6. 7.]]
#shape:(4, 2)
#axis=1时是行
#[[0. 1. 4. 5.]
# [2. 3. 6. 7.]]

#Tensor与Numpy转换
#Tensor转换为Numpy 与张量创建相同，使用asnumpy()将Tensor变量转换为Numpy变量
t = ops.ones(5, mindspore.float32)
print(f"t: {t}")   #t: [1. 1. 1. 1. 1.]
n = t.asnumpy()
print(f"n: {n}")   #n: [1. 1. 1. 1. 1.]

#Numpy转换为Tensor 使用Tensor()将Numpy变量转换为Tensor变量
n = np.ones(5)
t = Tensor.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")   #t: [2. 2. 2. 2. 2.]
print(f"n: {n}")   #n: [2. 2. 2. 2. 2.]











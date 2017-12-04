# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:51:01 2017

@author: leo
"""


import sys  
from caffe  import layers as L,params as P,to_proto

deploy='deploy.prototxt'    #文件保存路径

def create_deploy():
    #少了第一层，data层
    conv1=L.Convolution(name='conv1',bottom='data', kernel_size=5, stride=1,num_output=20, pad=0,weight_filler=dict(type='xavier'))
    pool1=L.Pooling(conv1,name='pool1',pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2=L.Convolution(pool1, name='conv2',kernel_size=5, stride=1,num_output=50, pad=0,weight_filler=dict(type='xavier'))
    pool2=L.Pooling(conv2, name='pool2',top='pool2', pool=P.Pooling.MAX, kernel_size=2, stride=2)
    fc3=L.InnerProduct(pool2, name='ip1',num_output=500,weight_filler=dict(type='xavier'))
    relu3=L.ReLU(fc3, name='relu1',in_place=True)
    fc4 = L.InnerProduct(relu3, name='ip2',num_output=10,weight_filler=dict(type='xavier'))
    #最后没有accuracy层，但有一个Softmax层
    prob=L.Softmax(fc4, name='prob')
    return to_proto(prob)
def write_deploy(): 
    with open(deploy, 'w') as f:
        f.write('name:"LeNet"\n')
        f.write('layer {\n')
        f.write('name:"data"\n')
        f.write('type:"Input"\n')
        f.write('input_param { shape : {')
        f.write('dim:1 ')
        f.write('dim:3 ')
        f.write('dim:28 ')
        f.write('dim:28 ')
        f.write('} }\n\n')
        f.write(str(create_deploy()))
if __name__ == '__main__':
    write_deploy()
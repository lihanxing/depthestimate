import os
import glob
import argparse
import matplotlib
import time
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt



# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
a = 1
filePath = args.input
for i, j, k in os.walk(filePath):
    if a == 1:
        a += 1
        continue

    path = r"result_test_input/"
    path = path + i[11:]
    i = i+"/color.jpg"
    print(path)
    print(i)
    os.makedirs(path)
    inputs = load_images( glob.glob(i) )
    temp = Image.open(i)
    ImgSize = temp.size
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)
    print(type(outputs))


    # output Images
    viz = display_images(outputs.copy())
    


    plt.figure(figsize=(5.13,5.12))
    plt.imshow(viz)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.axis('off')
    fig = plt.gcf()

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    plt.savefig(("%s"%path)+"/depth.png",bbox_inches='tight',pad_inches = 0)
    plt.clf()
    
    # 图片路径，相对路径
    image_path = path+"/depth.png"
    # 读取图片
    image = Image.open(image_path)
    # 输出维度
    print("RGB图像的维度：", np.array(image).shape)
    # 显示原图

    # RGB转换我灰度图像
    image_transforms = transforms.Compose([
        transforms.Grayscale(1)
    ])
    image = image_transforms(image)
    # 输出灰度图像的维度
    print("灰度图像维度： ", np.array(image).shape)
    # 显示灰度图像
    image.save(image_path)
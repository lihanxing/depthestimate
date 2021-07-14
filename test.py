import os
import glob
import argparse
import matplotlib
import time
from PIL import Image
import cv2
import numpy as np

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
parser.add_argument('--output',default='test_result', type=str, help='Output filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
temp = Image.open(args.input)
ImgSize = temp.size
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')

#按时间保存预测后图片
uuid_str = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
tmp_file_name ='%s.png' % uuid_str


# output Images
viz = display_images(outputs.copy())
# print(viz)
#
# viz = Image.fromarray(viz)

plt.figure(figsize=(5.13,5.12),dpi=100)
plt.imshow(viz)

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
fig = plt.gcf()

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)

plt.savefig(("%s"%args.output)+"/depth.png",bbox_inches='tight',pad_inches = 0)
# image_data = np.asarray(img)
# # cv2.imshow('image',image_data)
# cv2.imwrite('messigray.png',image_data)
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


image_path = "result1/3dfront-dataset-batch-split-10k-0-cam-0/depth.png"
image = Image.open(image_path)
 
input_transform = transforms.Compose([
   transforms.Grayscale(1), #这一句就是转为单通道灰度图像
   transforms.ToTensor(),
])
image_tensor = input_transform(image)
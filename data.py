import numpy as np
from utils import DepthNorm, load_images
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy



#================
# Real dataset
#================

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

# 这里的功能是对nyu图片进行缩放
def cmp_resize(img, resolution=512, padding=6):
    # skimage.transform实现图片缩放与形变
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution)), preserve_range=True, mode='reflect', anti_aliasing=True )


def get_cmp_data(batch_size):
    data = extract_zip("dataset.zip")

    cmp2_train_input = open("train.csv","r")
    cmp2_train_file = cmp2_train_input.read()
    cmp2_train = list((row.split(',') for row in (cmp2_train_file).split('\n') if len(row) > 0))

    cmp2_test_input = open("test.csv","r")
    cmp2_test_file = cmp2_test_input.read()
    cmp2_test = list((row.split(',') for row in (cmp2_test_file).split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 512, 512, 3)
    shape_depth = (batch_size, 256, 256, 1)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, cmp2_train, cmp2_test, shape_rgb, shape_depth

#这个函数的命名不变，以便train.py调用
def get_nyu_train_test_data(batch_size):
    data, cmp2_train, cmp2_test, shape_rgb, shape_depth = get_cmp_data(batch_size)

    train_generator = CMP_BasicAugmentRGBSequence(data, cmp2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = CMP_BasicRGBSequence(data,cmp2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

class CMP_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(512,512,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(512,512,1)/255*self.maxDepth,0,self.maxDepth)
            
            # y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = cmp_resize(x, 512)
            batch_y[i] = cmp_resize(y, 256)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class CMP_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(512,512,3)/255,0,1)
            y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(512,512,1).copy().astype(float) / 10.0
            # y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = cmp_resize(x, 512)
            batch_y[i] = cmp_resize(y, 256)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

#================
# Unreal dataset
#================

import cv2
from skimage.transform import resize

def get_unreal_data(batch_size):
    shape_rgb = (batch_size, 512, 512, 3)
    shape_depth = (batch_size, 512, 512, 1)

    # Open data file
    
    data = extract_zip("dataset.zip")

    cmp2_train_input = open("train.csv","r")
    cmp2_train_file = cmp2_train_input.read()
    cmp2_train = list((row.split(',') for row in (cmp2_train_file).split('\n') if len(row) > 0))

    cmp2_test_input = open("test.csv","r")
    cmp2_test_file = cmp2_test_input.read()
    cmp2_test = list((row.split(',') for row in (cmp2_test_file).split('\n') if len(row) > 0))

    # Helpful for testing...
    if False:
        unreal_train = unreal_train[:10]
        unreal_test = unreal_test[:10]

    return data, cmp2_train, cmp2_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)
    
    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        
        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            
            ##cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像
            # rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            # depth_sample = self.data['y/{}'.format(sample)] 
            # depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )
            
            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) ))/255, 0, 1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(512,512,), 10, self.maxDepth)
            # y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])
                
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y
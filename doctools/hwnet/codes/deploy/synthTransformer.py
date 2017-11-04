import torch
import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io, transform

#import pdb

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt, label, roi = sample['image'], sample['gt'], sample['label'], sample['roi']
        
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image,(new_w, new_h))
        
        return {'image': img, 'gt': gt, 'label': label, 'roi': roi}

class ElasticTransformation(object):
    """ElasticTransformation the image in a sample to a given size.
    Code adapted from https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

    Args:
        
    """
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, sample):
        
        image, gt, label, roi = sample['image'], sample['gt'], sample['label'], sample['roi']
        
        #calling elastic distortion only prob% times
        if(np.random.rand()<self.prob):
            return {'image': image, 'gt': gt, 'label': label, 'roi': roi}
        
        #print('Elastic Distortion')
        alpha = image.shape[0]*0.8
        sigma = image.shape[0]*0.08
        alpha_affine = image.shape[0]*0.009 
        random_state = None
       
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, 
                                                         center_square[1]-square_size], 
                                                        center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, 
                                           alpha_affine, 
                                           size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)) 
        
        newimage = map_coordinates(image, indices, order=1, mode='constant', 
                                         cval=255).reshape(shape)
        
        cords = np.where(newimage!=255)
        if cords[0].size!=0:
            roi = np.asarray([0.0, min(cords[1]),min(cords[0]),max(cords[1]),max(cords[0])],dtype=np.float32)
        else:
            print('Warning in reading ROI')
            roi = np.asarray([0.0, 0.0,0.0,newimage.shape[1],newimage.shape[0]],dtype=np.float32)
        
        return {'image': newimage, 'gt': gt, 'label': label, 'roi': roi}

class AffineTransformation(object):
    """AffineTransformation
    """
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, sample):
        image, gt, label, roi = sample['image'], sample['gt'], sample['label'], sample['roi']
        
        #calling affine distortion only prob% times
        if(np.random.rand()<self.prob):
            return {'image': image, 'gt': gt, 'label': label, 'roi': roi}
        
        rows, cols = image.shape

        if(np.random.rand()<0.5):
            #Rotation
            randAngles = range(-5,15,1)
            rotAngle = randAngles[np.random.randint(len(randAngles))]

            height, width = image.shape[:2]
            image_center = (width/2, height/2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, rotAngle, 1.)

            abs_cos = abs(rotation_mat[0,0])
            abs_sin = abs(rotation_mat[0,1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]

            image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            #print('Rotation with %d angles'%(rotAngle))
        else:
            #Shearing
            shearAngle=-0.5+np.random.rand();
            M = np.array([[1.0,shearAngle,0.0],[0.0,1.0,0.0]])
            image = cv2.warpAffine(image, M, (cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            #print('Shear with %f angles'%(shearAngle))

        #Padding
        if(np.random.rand()<0.5):
            #print('Padding')
            padTop = np.random.randint(20)
            padBottom = np.random.randint(20)
            padLeft = np.random.randint(20)
            padRight = np.random.randint(20)
            image = cv2.copyMakeBorder(image,padTop,padBottom,
                    padLeft,padRight, cv2.BORDER_CONSTANT, value=255)
        
        image = cv2.resize(image,(cols,rows))
        
        cords = np.where(image!=255)
        if cords[0].size!=0:
            roi = np.asarray([0.0, min(cords[1]),min(cords[0]),max(cords[1]),max(cords[0])],dtype=np.float32)
        else:
            print('Warning in reading ROI')
            roi = np.asarray([0.0, 0.0,0.0,image.shape[1],image.shape[0]],dtype=np.float32)

        return {'image': image, 'gt': gt, 'label': label, 'roi': roi}

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, roi, gt = sample['image'], sample['label'], sample['roi'], sample['gt']
        #pdb.set_trace()
        #roiImage = image[int(roi[2]):int(roi[4]),int(roi[1]):int(roi[3])]
        #roiImage = (roiImage-np.mean(roiImage)) / ((np.std(roiImage) + 0.0001) / 128.0)
        
        #ToD0: Check whether np.max should be given or some pre-defined max value
        #image = np.ones(image.shape,dtype=np.float32)*np.max(roiImage)
        #image[int(roi[2]):int(roi[4]),int(roi[1]):int(roi[3])] = roiImage
        image = (image-np.mean(image)) / ((np.std(image) + 0.0001) / 128.0)

        return {'image': image,
                'label': label,
                'roi': roi,
                'gt': gt}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, roi = sample['image'], sample['label'], sample['roi']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'roi': torch.from_numpy(roi)}
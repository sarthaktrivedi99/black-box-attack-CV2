import numpy as np
import tensorflow as tf
import os
import random
from typing import Tuple
from tensorflow.keras.preprocessing import image
from scipy.fftpack import idct,dct

class Image_Loader:
    def __init__(self,location,batch_size,image_size,targeted=False):
        self.location = location
        self.batch_size = batch_size
        self.image_size = image_size
        self.curr_index = 0
        self.targeted = targeted
        self.get_image_locations()
        self.batch_x = []
        self.batch_y = []

    def shuffle_location(self,list_location,labels)->Tuple[list,list]:
        """
        Shuffles the list of locations and labels, so that the pertrubations can be general.
        :param list_location: list of location of images
        :param labels: their corresponding labels
        :return: list_location and labels
        """
        zipped = list(zip(list_location,labels))
        random.shuffle(zipped)
        list_location,labels = zip(*zipped)
        return list_location,labels

    def get_image_locations(self)->None:
        """
        Gets the file locations of all images in the dataset.
        """
        self.list_locations = []
        self.labels = []
        final_path = lambda i, x: os.path.join('.', self.location, i, x)
        targeted_label = set(os.listdir(self.location))
        for i in os.listdir(self.location):
            if(os.path.isdir(os.path.join(self.location,i))):
                for j in os.listdir(os.path.join(self.location,i)):
                    self.list_locations.append(final_path(i,j))
                    if self.targeted:
                        self.labels.append(np.random.choice(targeted_label))
                    else:
                        self.labels.append(i)
        self.list_locations,self.labels = self.shuffle_location(self.list_locations,self.labels)

    def __len__(self):
        return len(self.list_locations)//self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        """
        Loads ImageNet images from a specified folder into batches.
        """
        if self.curr_index==len(self.list_locations):
            self.curr_index=0
        if self.batch_size==1:
            img,label = image.img_to_array(image.load_img(self.list_locations[self.curr_index],target_size = (self.image_size, self.image_size))),self.labels[self.curr_index]
            self.curr_index+=self.batch_size
            return img,label
        self.batch_y = self.labels[self.curr_index:self.curr_index+self.batch_size]
        for i in self.list_locations[self.curr_index:self.curr_index+self.batch_size]:
            self.batch_x.append(image.img_to_array(image.load_img(i,target_size = (self.image_size, self.image_size))))
        self.curr_index+=self.batch_size
        return np.asarray(self.batch_x,dtype=object),np.asarray(self.batch_y,dtype=object)


def block_idct(x, block_size=8, linf_bound=0.0):
    """
    produces the inverse DCT image for a DCT image
    :param x: input image in DCT basis
    :param block_size: size of blocks if the image has to be processed in smaller blocks
    :param linf_bound: constraint over output image
    :return: an image in original basis (this basis can be anything pixel basis or any other kind of basis)
    """
    z = np.zeros_like(x)
    num_blocks = int(x.shape[2] / block_size)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if len(x.shape)>3:
                submat = x[:, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size),:]
            # Turns out applying iDCT to two times on different axis is the correct way to apply it on an image
            # https://stackoverflow.com/questions/7110899/how-do-i-apply-a-dct-to-an-image-in-python
            z[:, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size),:] = idct(idct(submat, axis=2, norm='ortho'), axis=1, norm='ortho')
    if linf_bound > 0:
        return z.clip(-linf_bound, linf_bound)
    else:
        return z
if __name__ == '__main__':
    images = Image_Loader(os.path.join('.','imagenette2','train'),1,244)
    for i in range(images.__len__()):
        img,label = next(images)

import numpy as np
from PIL import Image
import os
import random
from typing import Tuple
from tensorflow.keras.preprocessing import image


class Image_Loader:
    def __init__(self,location,batch_size,image_size):
        self.location = location
        self.batch_size = batch_size
        self.image_size = image_size
        self.curr_index = 0
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
        for i in os.listdir(self.location):
            if(os.path.isdir(os.path.join(self.location,i))):
                for j in os.listdir(os.path.join(self.location,i)):
                    self.list_locations.append(final_path(i,j))
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
            return image.img_to_array(image.load_img(self.list_locations[self.curr_index],target_size = (self.image_size, self.image_size))),self.labels[self.curr_index]
        self.batch_y = self.labels[self.curr_index:self.curr_index+self.batch_size]
        for i in self.list_locations[self.curr_index:self.curr_index+self.batch_size]:
            self.batch_x.append(image.img_to_array(image.load_img(i,target_size = (self.image_size, self.image_size))))
        self.curr_index+=self.batch_size
        return np.asarray(self.batch_x,dtype=object),np.asarray(self.batch_y,dtype=object)



if __name__ == '__main__':
    images = Image_Loader(os.path.join('.','imagenette2','train'),20,244)
    # for i in tqdm(range(images.__len__())):
    #     next(images)[0].shape
    # print(next(images)[0][0])
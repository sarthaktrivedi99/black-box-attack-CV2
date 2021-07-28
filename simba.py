from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from  utils import Image_Loader
from keras.applications.resnet_v2 import preprocess_input,decode_predictions
from tqdm import tqdm

class SimBA:
    def __init__(self,model,preprocess_fn,decode_fn):
        """
        SimBA class constructor.
        :param model: Tensorflow model that you want to perform the attack on
        :param preprocess_fn: default models have preprocessing functions to appropriately preprocess inputs
        :param decode_fn: default models have decoding function to convert probs to outputs with labels
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.decode_fn = decode_fn

    def get_probs(self,x)-> np.array:
        """
        Uses the model to generate probability scores for the given input i.e. P(y|x).
        :param x: image tensor
        :return: list of probabilities
        """
        if len(x.shape)==3:
            x = x[np.newaxis,...]
        return self.model.predict(x)

    def decode_probs_custom(self,probs,y)->list:
        """
        Custom probability decoding function. Takes the probability and generates a custom list of dicts
        for each input image that contains the label and the prediction score and returns the appropriate
        probability for the given class label.
        :param probs: list of probabilities
        :return: list of scores
        """
        prob_dict = [{j[0]: j[-1] for j in i} for i in self.decode_fn(probs, top=1000)]
        if len(prob_dict)==1:
            return prob_dict[0][y]
        else:
            probs = [prob_dict[i][y[i]] for i in range(prob_dict)]
            return probs

    def simba_single(self,x,y,iters=1000,epsilon=0.04):
        n_dim = tf.reshape(x,(1,-1)).shape.as_list()[-1]
        perm = tf.range(n_dim)
        perm = tf.random.shuffle(perm)
        x = tf.expand_dims(x,0)
        petrubation = np.zeros_like(x)
        last_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(x)),y)
        for i in range(iters):
            diff = np.zeros([n_dim])
            diff[perm[i]] = epsilon*255
            left_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(tf.clip_by_value(x - (diff.reshape(x.shape.as_list())),0,255))),y)
            if (left_prob <= last_prob):
                x = tf.clip_by_value(x - (diff.reshape(x.shape.as_list())),0,255)
                last_prob = left_prob
            else:
                right_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(tf.clip_by_value(x + (diff.reshape(x.shape.as_list())),0,255))),y)
                if (right_prob <= last_prob):
                    x = tf.clip_by_value(x + (diff.reshape(x.shape.as_list())),0,255)
                    last_prob = right_prob
            petrubation += np.clip(diff.reshape(x.shape.as_list()),0,255)
            if i % 100 == 0:
                print(f'Iteration - {i} ; Confidence - {last_prob}')
        return np.squeeze(petrubation,0)

if __name__ == '__main__':
    model = keras.applications.ResNet50V2(weights='imagenet')
    input_shape = model.input.shape[1]
    x,y = Image_Loader(os.path.join('.', 'imagenette2', 'val'),1,input_shape).__next__()
    perturbation = SimBA(model,preprocess_input,decode_predictions).simba_single(x,y)

from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from  utils import Image_Loader,block_idct
from typing import Tuple
from keras.applications.resnet_v2 import preprocess_input,decode_predictions

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
    def most_prob_class(self,probs):
        """
        Returns the output class given model's output probability
        :param probs: input the probability output from the model
        :return: the output class of the model given the input
        """
        return self.decode_fn(probs, top=1)[0][0][0]
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

    def simba_single(self,x,y,iters=1000,epsilon=0.2,attack_mode='pixel',targeted=False,log_every=100) -> Tuple[np.array,bool,int,int]:
        """
        Runs the SimBA attack on single image. Takes in a perturbation and returns an updated perturbation matrix.
        :param x: Input image
        :param y: label
        :param attack_mode: attack mode based on basis in which attack takes place
        :param iters: number of iterations to run this attack
        :param log_every: intervals at which logs take place
        :param epsilon: epsilon value (controls how much perturbation is added)
        :param targeted: if the attack is targeted or untargeted
        :return: Final image, success, number of queries required for this image and L2 norm
        """
        assert attack_mode in ['pixel','dct']
        n_dim = tf.reshape(x,(1,-1)).shape.as_list()[-1]
        # The perm matrix holds the
        perm = tf.range(n_dim)
        perm = tf.random.shuffle(perm)
        queries = 0
        x = tf.expand_dims(x,0)
        perturbation = np.zeros_like(x,dtype=np.float)
        perturbation = np.expand_dims(perturbation,0)
        last_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(x)),y)
        for i in range(iters):
            diff = np.zeros([n_dim])
            # Since we attack on a image that is in the range [0,255] we multiply the epsilon with 255 to scale it.
            diff[perm[i]] = epsilon*255
            if attack_mode == 'dct':
                # directly making changes in DCT space and then converting the changes to pixel space
                diff = np.ndarray.flatten(np.clip(block_idct(diff.reshape(x.shape.as_list())),0,255))
            left_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(tf.clip_by_value(x - (diff.reshape(x.shape.as_list())),0,255))),y)
            queries+=1
            # This if condition forces the opposite condition(i.e. moving towards a larger value for the targeted label)
            if targeted != (left_prob < last_prob):
                x = tf.clip_by_value(x - (diff.reshape(x.shape.as_list())),0,255)
                last_prob = left_prob
            else:
                right_prob = self.decode_probs_custom(self.get_probs(self.preprocess_fn(tf.clip_by_value(x + (diff.reshape(x.shape.as_list())),0,255))),y)
                queries += 1
                if targeted != (right_prob < last_prob):
                    x = tf.clip_by_value(x + (diff.reshape(x.shape.as_list())),0,255)
                    last_prob = right_prob
            perturbation += np.clip(diff.reshape(x.shape.as_list()),0,255)
            if i % 100 == 0:
                print(f'Iteration - {i} ; Confidence - {last_prob}')
                if targeted:
                    if self.most_prob_class(self.get_probs(self.preprocess_fn(x)))==y:
                        print(f'Predicted class -  {self.most_prob_class(self.get_probs(self.preprocess_fn(x)))},Given class - {y},stopping early...')
                        return np.squeeze(x,0),True,queries,np.linalg.norm(perturbation)
                elif self.most_prob_class(self.get_probs(self.preprocess_fn(x)))!=y:
                    print(f'Predicted class -  {self.most_prob_class(self.get_probs(self.preprocess_fn(x)))},Given class - {y},stopping early...')
                    return np.squeeze(x,0),True,queries,np.linalg.norm(perturbation)

        return np.squeeze(x,0), False, queries, np.linalg.norm(perturbation)


if __name__ == '__main__':
    model = keras.applications.ResNet50V2(weights='imagenet')
    input_shape = model.input.shape[1]
    # x,y = Image_Loader(os.path.join('.', 'imagenette2', 'val'),1,input_shape).__next__()
    sim = SimBA(model,preprocess_input,decode_predictions)
    x = tf.random.uniform(shape=[15,3*14*14])
    expanded = sim.expand_vector(x,14,input_shape)
    transformed = block_idct(expanded, block_size=input_shape, linf_bound=0.0)
    print(np.max(transformed))

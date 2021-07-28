from simba import SimBA
import os
from utils import Image_Loader
from tensorflow.keras.applications import ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,VGG16,VGG19
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=10000, help='maximum number of iterations')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
args = parser.parse_args()

models = {'resnet50':ResNet50(weights='imagenet'),
          'resnet101':ResNet101(weights='imagenet'),
          'resnet151':ResNet152(weights='imagenet'),
          'resnet50v2':ResNet50V2(weights='imagenet'),
          'resnet101v2':ResNet101V2(weights='imagenet'),
          'resnet152v2':ResNet152V2(weights='imagenet'),
          'vgg16':VGG16(weights='imagenet'),
          'vgg19':VGG19(weights='imagenet')}

if args.model[-2:]=='v2':
    from keras.applications.resnet_v2 import preprocess_input,decode_predictions
elif args.model=='vgg16':
    from keras.applications.vgg16 import preprocess_input, decode_predictions
elif args.model=='vgg19':
    from keras.applications.vgg19 import preprocess_input, decode_predictions
else:
    from keras.applications.resnet import preprocess_input, decode_predictions

model = models[args.model]
input_shape = model.input.shape[1]
x,y = Image_Loader(os.path.join(args.data_root, 'val'),args.batch_size,input_shape).__next__()
sim = SimBA(model,preprocess_input,decode_predictions)
petrubation = sim.simba_single(x,y,iters=1000)
plt.imshow(np.clip(x+petrubation,0,255)/255)
plt.show()
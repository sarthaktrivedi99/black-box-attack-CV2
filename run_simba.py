from simba import SimBA
import os
from utils import Image_Loader
from tensorflow.keras.applications import ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,VGG16,VGG19
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--attack_mode', type=str, default='dct', help='type of base model to use')
parser.add_argument('--output', type=str, default='save', help='number of image samples')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
args = parser.parse_args()
if not os.path.isdir(os.path.join('.',args.output)):
    os.mkdir(args.output)
    os.mkdir(os.path.join(args.output,'original'))
    os.mkdir(os.path.join(args.output,'modified'))
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
perturbation = np.zeros(shape=(input_shape,input_shape,3))
loader = Image_Loader(os.path.join(args.data_root, 'val'),args.batch_size,input_shape)
sim = SimBA(model,preprocess_input,decode_predictions)
succ_log = np.zeros(shape=len(loader),dtype=np.bool)
query_log = np.zeros(shape=len(loader),dtype=np.float)
l2_norm_log = np.zeros(shape=len(loader),dtype=np.float)
for i in range(len(loader)):
    x,y = next(loader)
    x_,succ,query,l2_norm = sim.simba_single(x,y,iters=args.num_runs,epsilon=args.epsilon,attack_mode=args.attack_mode,log_every=args.log_every)
    print(succ,query,l2_norm)
    succ_log[i],query_log[i],l2_norm_log[i] = succ, query, l2_norm
    plt.imsave(os.path.join('.',args.output,'original',f'{i}.jpg'),x /255)
    plt.imsave(os.path.join('.', args.output, 'modified', f'{i}.jpg'), x_ / 255)
with open(os.path.join('.',args.output,'data.json'),'w') as f:
    f.write(json.dumps({'succ_log':succ_log.tolist(),'query_log':query_log.tolist(),'l2_norm_log':l2_norm_log.tolist()}))
    f.close()
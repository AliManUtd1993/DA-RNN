import _init_paths
from fcn.test import test_net
from fcn.test import test_net_single_frame
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp
import _init_paths
from fcn.test import test_net
from fcn.test import test_net_single_frame
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname('/home/aliman/DA-RNN-master/tools/_init_paths.py')

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
################# WE MIGHT HAVE ERROR RELATED TO PATH, IT CAN BE EASILY FIXED LATER

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='pretrained model',
                        default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='shapenet_scene_val', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
                        default=None, type=str)
    parser.add_argument('--kfusion', dest='kfusion',
                        help='run kinect fusion or not',
                        default=False, type=bool)

    args = parser.parse_args(['--gpu', '0', '--network', 'vgg16',  \
                              '--model',\
                              '/home/aliman/DA-RNN-master/data/fcn_models/rgbd_scene/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000.ckpt',\
                              '--weights', '/home/aliman/DA-RNN-master/data/imagenet_models/vgg16_convs.npy', \
                             '--imdb', 'rgbd_scene_train', '--cfg', \
                              '/home/aliman/DA-RNN-master/experiments/cfgs/rgbd_scene_multi_rgbd.yml'])
    return args
args = parse_args()

print('Called with args:')
print(args)


args = parse_args()

print('Called with args:')
print(args)

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

print('Using config:')
pprint.pprint(cfg)

weights_filename = 'vgg16_convs.npy'#os.path.splitext(os.path.basename(args.model))[0]

imdb = get_imdb(args.imdb_name)

cfg.GPU_ID = args.gpu_id
device_name = '/gpu:{:d}'.format(args.gpu_id)
print device_name

cfg.TRAIN.NUM_STEPS = 1
cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
if cfg.NETWORK == 'FCN8VGG':
    path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
    cfg.TRAIN.MODEL_PATH = path
cfg.TRAIN.TRAINABLE = False

from networks.factory import get_network
network = get_network(args.network_name)
print 'Use network `{:s}` in training'.format(args.network_name)


net=network
x=net.get_output('score_conv5')
#x.shape
theLayerWeWant_rs=tf.image.resize_images(x,[8,8])
theLayerWeWant_f=tf.contrib.layers.flatten(theLayerWeWant_rs)
#theLayerWeWant_f.shape  
#  TensorShape([Dimension(None), Dimension(4096)])

def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        # matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
        #                          tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
            
            
           
batch_size = theLayerWeWant_f.shape[0]
####################    3: number of scence classes
scene_score = linear(theLayerWeWant_f, 3) ####################
predict_scene = tf.argmax(scene_score, axis=1)
y = tf.placeholder("float", shape=[None, 3])## REAL LABELS, REMEMBER TO LOAD THEM WHEN THE DATA IS READY

cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scene_score))

t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
wanted_vars = [var for var in t_vars if 'Linear' in var.name]

updates = tf.train.GradientDescentOptimizer(0.01).minimize(loss = cost, var_list = wanted_vars)



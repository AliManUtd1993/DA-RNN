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
from gt_data_layer.layer import GtDataLayer
from gt_single_data_layer.layer import GtSingleDataLayer
from utils.timer import Timer
import numpy as np
import threading

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
roidb = get_training_roidb(imdb)
output_dir = get_output_dir(imdb, None)
cfg.GPU_ID = args.gpu_id
device_name = '/gpu:{:d}'.format(args.gpu_id)
print device_name
max_iters=150
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

loss    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scene_score))

t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
wanted_vars = [var for var in t_vars if 'Linear' in var.name]
learning_rate=0.01
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss = loss, var_list = wanted_vars)


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        # For checkpoint
        self.saver = tf.train.Saver()


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + 'ours.ckpt')##remember to change the name in our test code based on this line
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_model(self, sess, train_op, loss, learning_rate, max_iters):
        """Network training loop."""
        # add summary
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        tf.get_default_graph().finalize()

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            summary, loss_value, lr, _ = sess.run([merged, loss, learning_rate, train_op])
            train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, lr: %.8f, time: %.2f' %\
                    (iter+1, max_iters, loss_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


    

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb


def load_and_enqueue(sess, net, roidb, num_classes, coord):
    if cfg.TRAIN.SINGLE_FRAME:
        # data layer
        data_layer = GtSingleDataLayer(roidb, num_classes)
    else:
        # data layer
        data_layer = GtDataLayer(roidb, num_classes)

    while not coord.should_stop():
        blobs = data_layer.forward()

        if cfg.INPUT == 'RGBD':
            data_blob = blobs['data_image_color']
            data_p_blob = blobs['data_image_depth']
        elif cfg.INPUT == 'COLOR':
            data_blob = blobs['data_image_color']
        elif cfg.INPUT == 'DEPTH':
            data_blob = blobs['data_image_depth']
        elif cfg.INPUT == 'NORMAL':
            data_blob = blobs['data_image_normal']

        if cfg.TRAIN.SINGLE_FRAME:
            if cfg.INPUT == 'RGBD':
                ##REMEMBER TO FEED SCENE LABEL TO y PLACE HOLDER LATER
                feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}

            else:
                
                feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}
        else:
            if cfg.INPUT == 'RGBD':
                feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], \
                           net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                           net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}
            else:
                feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], \
                           net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                           net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}

        sess.run(net.enqueue_op, feed_dict=feed_dict)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

    # thread to load data
    coord = tf.train.Coordinator()
    t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
    t.start()
    print 'Solving...'
    sw.train_model(sess, train_op, loss, learning_rate, max_iters)
    print 'done solving'

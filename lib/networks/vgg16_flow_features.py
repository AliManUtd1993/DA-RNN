import tensorflow as tf
from networks.network import Network
from fcn.config import cfg
zero_out_module = tf.load_op_library('lib/triplet_flow_loss/triplet_flow_loss.so')


class vgg16_flow_features(Network):
    def __init__(self, input_format, num_classes, num_units, scales, vertex_reg=False, trainable=True):
        self.inputs = []
        self.input_format = input_format
        self.num_output_dimensions = 2  # formerly num_classes
        self.num_units = num_units
        self.scale = 1 / scales[0]
        self.vertex_reg = vertex_reg

        self.data_left = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data_right = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.gt_flow = tf.placeholder(tf.float32, shape=[None, None, None, self.num_output_dimensions])
        self.occluded = tf.placeholder(tf.int32, shape=[None, None, None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        self.q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32])
        self.enqueue_op = self.q.enqueue([self.data_left, self.data_right, self.gt_flow, self.occluded, self.keep_prob])
        data_left, data_right, gt_flow, occluded, self.keep_prob_queue = self.q.dequeue()
        self.layers = dict({'data_left': data_left, 'data_right': data_right, 'gt_flow': gt_flow, 'occluded': occluded})

        self.close_queue_op = self.q.close(cancel_pending_enqueues=True)
        self.queue_size_op = self.q.size('queue_size')
        self.trainable = trainable
        self.setup()

    def setup(self):
        trainable = True
        reuse = True
        (self.feed('data_left')
             .add_immediate(tf.constant(0.0, tf.float32), name='data_left_tap')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=trainable)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=trainable)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=trainable)
             .add_immediate(tf.constant(0.0, tf.float32), name='conv4_3_l')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=trainable)
             .add_immediate(tf.constant(0.0, tf.float32), name='conv5_3_l'))

        (self.feed('data_right')
             .add_immediate(tf.constant(0.0, tf.float32), name='data_right_tap')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=trainable, reuse=reuse)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=trainable, reuse=reuse)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=trainable, reuse=reuse)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=trainable, reuse=reuse)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=trainable, reuse=reuse)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=trainable, reuse=reuse)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=trainable, reuse=reuse)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=trainable, reuse=reuse)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=trainable, reuse=reuse)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=trainable, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='conv4_3_r')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=trainable, reuse=reuse)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=trainable, reuse=reuse)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=trainable, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='conv5_3_r'))

        (self.feed(['conv5_3_l', 'conv5_3_r', 'gt_flow', 'occluded'])
            .triplet_flow_loss(margin=1, name="triplet_flow_loss_name"))
        pass



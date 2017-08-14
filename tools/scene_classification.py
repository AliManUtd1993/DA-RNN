import tensorflow as tf

#remember to change the directory later
saver = tf.train.import_meta_graph('../output/rgbd_scene/rgbd_scene_train/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000.ckpt.meta')

with tf.session as sess:
  new_saver = tf.train.import_meta_graph('../output/rgbd_scene/rgbd_scene_train/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))

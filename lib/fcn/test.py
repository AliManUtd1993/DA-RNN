# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an imdb (image database)."""

from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
from utils.blob import im_list_to_blob, pad_im, unpad_im
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import scipy.io
import time
from normals import gpu_normals
# from pose_estimation import ransac
#from kinect_fusion import kfusion
from utils import sintel_utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
from gt_flow_data_layer.layer import GtFlowDataLayer
from gt_lov_correspondence_layer.layer import GtLOVFlowDataLayer
import scipy.ndimage

import triplet_flow_loss.run_slow_flow_calculator_process
# import pyximport; pyximport.install()
import utils.plotting_tools

# output/pupper_dataset/batch_size_2_loss_L2_optimizer_ADAM_skip_link_1_True_2_True_3_True_2017-06-29/vgg16_flow_sintel_albedo_iter_16000.ckpt
#output/sintel_albedo_small_training_set_fewer_skiplinks_trainable/batch_size_4_loss_L2_optimizer_ADAM_skip_link_1_False_2_False_3_True_2017-07-06/vgg16_flow_sintel_albedo_iter_9000.ckpt


def _get_image_blob(im, im_depth, meta_data):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    # RGB
    im_orig = im.astype(np.float32, copy=True)
    # mask the color image according to depth
    if cfg.EXP_DIR == 'rgbd_scene':
        I = np.where(im_depth == 0)
        im_orig[I[0], I[1], :] = 0

    processed_ims_rescale = []
    im_scale = cfg.TEST.SCALES_BASE[0]
    im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_rescale.append(im_rescale)

    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # depth
    im_orig = im_depth.astype(np.float32, copy=True)
    im_orig = im_orig / im_orig.max() * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # meta data
    K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # normals
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
    nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
    im_normal = 127.5 * nmap + 127.5
    im_normal = im_normal.astype(np.uint8)
    im_normal = im_normal[:, :, (2, 1, 0)]
    im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

    im_orig = im_normal.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_normal = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_normal.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)
    blob_normal = im_list_to_blob(processed_ims_normal, 3)

    return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


######################
# test single frame(?)
######################
def im_segment_single_frame(sess, net, im, im_depth, meta_data, num_classes):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

    # use a fake label blob of ones
    height = im_depth.shape[0]
    width = im_depth.shape[1]
    label_blob = np.ones((1, height, width, num_classes), dtype=np.float32)

    if cfg.TEST.GAN:
        gan_label_true_blob = np.zeros((1, height / 32, width / 32, 2), dtype=np.float32)
        gan_label_false_blob = np.zeros((1, height / 32, width / 32, 2), dtype=np.float32)
        gan_label_color_blob = np.zeros((num_classes, 3), dtype=np.float32)

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        if cfg.TEST.GAN:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.gan_label_true: gan_label_true_blob, net.gan_label_false: gan_label_false_blob, net.gan_label_color: gan_label_color_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.GAN:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.gan_label_true: gan_label_true_blob, net.gan_label_false: gan_label_false_blob, net.gan_label_color: gan_label_color_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.NETWORK == 'FCN8VGG':
        labels_2d, probs = sess.run([net.label_2d, net.prob], feed_dict=feed_dict)
    else:
        if cfg.TEST.VERTEX_REG:
            labels_2d, probs, vertex_pred = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred')], feed_dict=feed_dict)
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')], feed_dict=feed_dict)
            vertex_pred = []

    return labels_2d[0,:,:].astype(np.uint8), probs[0,:,:,:], vertex_pred


def im_segment(sess, net, im, im_depth, state, weights, points, meta_data, voxelizer, pose_world2live, pose_live2world):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])

    # construct the meta data
    """
    format of the meta_data
    intrinsic matrix: meta_data[0 ~ 8]
    inverse intrinsic matrix: meta_data[9 ~ 17]
    pose_world2live: meta_data[18 ~ 29]
    pose_live2world: meta_data[30 ~ 41]
    voxel step size: meta_data[42, 43, 44]
    voxel min value: meta_data[45, 46, 47]
    """
    K = np.matrix(meta_data['intrinsic_matrix'])
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    mdata[18:30] = pose_world2live.flatten()
    mdata[30:42] = pose_live2world.flatten()
    mdata[42] = voxelizer.step_x
    mdata[43] = voxelizer.step_y
    mdata[44] = voxelizer.step_z
    mdata[45] = voxelizer.min_x
    mdata[46] = voxelizer.min_y
    mdata[47] = voxelizer.min_z
    if cfg.FLIP_X:
        mdata[0] = -1 * mdata[0]
        mdata[9] = -1 * mdata[9]
        mdata[11] = -1 * mdata[11]

    # construct blobs
    height = im_depth.shape[0]
    width = im_depth.shape[1]
    depth_blob = np.zeros((1, height, width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    depth_blob[0,:,:,0] = depth
    meta_data_blob[0,0,0,:] = mdata
    # use a fake label blob of 1s
    label_blob = np.ones((1, height, width, voxelizer.num_classes), dtype=np.float32)

    # reshape the blobs
    num_steps = 1
    ims_per_batch = 1
    height_blob = im_blob.shape[1]
    width_blob = im_blob.shape[2]
    im_blob = im_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))
    im_depth_blob = im_depth_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))
    im_normal_blob = im_normal_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))

    label_blob = label_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    depth_blob = depth_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    meta_data_blob = meta_data_blob.reshape((num_steps, ims_per_batch, 1, 1, -1))

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
        data_p_blob = None
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
        data_p_blob = None
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob
        data_p_blob = None
    else:
        data_blob = None
        data_p_blob = None

    if cfg.INPUT == 'RGBD':
        feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.state: state, net.weights: weights, net.depth: depth_blob, \
                     net.meta_data: meta_data_blob, net.points: points, net.keep_prob: 1.0}
    else:
        feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.state: state, net.weights: weights, net.depth: depth_blob,
                     net.meta_data: meta_data_blob, net.points: points, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    labels_pred_2d, probs, state, weights, points = sess.run([net.get_output('labels_pred_2d'), net.get_output('probs'),
        net.get_output('output_state'), net.get_output('output_weights'), net.get_output('output_points')], feed_dict=feed_dict)

    labels_2d = labels_pred_2d[0]

    return labels_2d[0,:,:].astype(np.uint8), probs[0][0,:,:,:], state, weights, points


def vis_segmentations(im, im_depth, labels, labels_gt, colors):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(221)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')
    ax.autoscale()

    # show depth
    ax = fig.add_subplot(222)
    plt.imshow(im_depth)
    ax.set_title('input depth')
    ax.autoscale()

    # show class label
    ax = fig.add_subplot(223)
    plt.imshow(labels)
    ax.set_title('class labels')
    ax.autoscale()

    ax = fig.add_subplot(224)
    plt.imshow(labels_gt)
    ax.set_title('gt class labels')
    ax.autoscale()

    # show the 3D points
    '''
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(224, projection='3d')
    ax.set_aspect('equal')

    points = points[0,:,:,:].reshape((-1, 3))

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    index = np.where(np.isfinite(X))[0]
    perm = np.random.permutation(np.arange(len(index)))
    num = min(10000, len(index))
    index = index[perm[:num]]
    ax.scatter(X[index], Y[index], Z[index], c='r', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    '''

    fig.tight_layout()
    plt.show()


###################
# test flow
###################
def test_flow_net(sess, net, imdb, weights_filename, n_images=None, save_image=False, training_iter='',
                  calculate_EPE_all_data=False):

    if weights_filename is not None:
        output_dir = get_output_dir(imdb, weights_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    np.random.seed(10)
    roidb_ordering = np.random.permutation(np.arange(len(imdb.roidb)))
    if n_images is None:
        n_images = len(imdb.roidb)
    roidb_ordering = roidb_ordering[0:n_images]
    EPE_list = list()

    if cfg.INPUT == "LEFT_RIGHT_CORRESPONDENCE":
        data_layer = GtLOVFlowDataLayer(imdb.roidb, None, single=True)
    else:
        data_layer = GtFlowDataLayer(imdb.roidb, None, single=True)

    for i in range(n_images):

        # Get network outputs
        blobs = data_layer.forward()
        left_blob = blobs['left_image']
        right_blob = blobs['right_image']
        flow_blob = blobs['flow']
        depth_blob = blobs['depth']
        gt_flow = flow_blob[0]
        occluded_blob = blobs['occluded']

        index = roidb_ordering[i]
        images = imdb.roidb[index]

        network_inputs = {net.data_left: left_blob, net.data_right: right_blob,
                          net.gt_flow: np.zeros(list(right_blob.shape[:3]) + list([2]), dtype=np.float32),
                          net.occluded: np.zeros(list(right_blob.shape[:3]) + list([1]), dtype=np.int32),
                          net.keep_prob: 1.0}
        network_outputs = [net.get_output('gt_flow')]
        results = siphon_outputs_single_frame(sess, net, network_inputs, network_outputs)
        flow_arrays = list([results[0][0]])

        # network_inputs = {net.data_left: left_blob, net.data_right: right_blob,
        #                   net.gt_flow: np.zeros(list(right_blob.shape[:3]) + list([2]), dtype=np.float32),
        #                   net.occluded: np.zeros(list(right_blob.shape[:3]) + list([1]), dtype=np.int32), net.keep_prob: 1.0}
        # network_outputs = [net.get_output('features_1x_l'), net.get_output('features_1x_r'), net.get_output('gt_flow'),
        #                    net.get_output('final_triplet_loss'), net.get_output('occluded'),
        #                    net.get_output("features_4x_l"), net.get_output("features_16x_l"),
        #                    net.get_output("features_4x_r"), net.get_output("features_16x_r"),
        #                    net.get_output("occluded_4x"), net.get_output("occluded_16x"),]
        # results = siphon_outputs_single_frame(sess, net, network_inputs, network_outputs)

        # # Pyramidal flow calculation
        # left_pyramid = (np.squeeze(results[6]), np.squeeze(results[5]), np.squeeze(results[0]))
        # right_pyramid = (np.squeeze(results[8]), np.squeeze(results[7]), np.squeeze(results[1]))
        # occluded_pyramid = (np.squeeze(results[10]), np.squeeze(results[9]), np.squeeze(results[4]))
        # predicted_flow, feature_errors, flow_arrays = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid(left_pyramid, right_pyramid,
        #                                    occluded_pyramid, neighborhood_len_import=100, interpolate_after=True)
        # # predicted_flow = interpolate_flow(np.squeeze(results[0]), np.squeeze(results[1]), predicted_flow)


        # Larger features flow calculation
        # upscale_16x_l = scipy.ndimage.zoom(np.squeeze(results[6]), [16, 16, 1], order=1)
        # upscale_16x_r = scipy.ndimage.zoom(np.squeeze(results[8]), [16, 16, 1], order=1)
        # upscale_4x_l = scipy.ndimage.zoom(np.squeeze(results[5]), [4, 4, 1], order=1)
        # upscale_4x_r = scipy.ndimage.zoom(np.squeeze(results[7]), [4, 4, 1], order=1)
        # huge_features_l = np.dstack([np.squeeze(results[0]), upscale_4x_l, upscale_16x_l])
        # huge_features_r = np.dstack([np.squeeze(results[1]), upscale_4x_r, upscale_16x_r])
        # left_pyramid = (scipy.ndimage.zoom(huge_features_l, [1/8.0, 1/8.0, 1], order=1), huge_features_l)
        # right_pyramid = (scipy.ndimage.zoom(huge_features_r, [1/8.0, 1/8.0, 1], order=1), huge_features_r)
        # occluded_pyramid = (scipy.ndimage.zoom(np.squeeze(results[4]), [1/8.0, 1/8.0], order=1), np.squeeze(results[4]))
        # predicted_flow, feature_errors, flow_arrays = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid(left_pyramid, right_pyramid,
        #                                  occluded_pyramid, neighborhood_len_import=100, interpolate_after=True)
        # # predicted_flow = interpolate_flow(np.squeeze(results[0]), np.squeeze(results[1]), predicted_flow)
        #
        # # predicted_flow, feature_errors, flow_arrays = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid([np.squeeze(results[0])], [np.squeeze(results[1])],
        # #                                    [np.squeeze(results[4])], neighborhood_len_import=200, interpolate_after=False)
        #
        # predicted_flow_cropped = predicted_flow[:gt_flow.shape[0], :gt_flow.shape[1]]
        # average_EPE = sintel_utils.calculate_EPE(gt_flow, predicted_flow_cropped)
        # zero_prediction_EPE = sintel_utils.calculate_EPE(gt_flow, np.zeros(gt_flow.shape))

        if calculate_EPE_all_data:
            path_segments = str(images['image_left']).split("/")
            print ("%3i / %i EPE is %7.4f for " % (i + 1, n_images, average_EPE)) + path_segments[-3] + "/" + path_segments[-2] + "/" + path_segments[-1]
            print "\tcalculated triplet loss is %7.4f" % float(results[3][0])
            EPE_list.append(average_EPE)
        else:
            fig = plt.figure()
            # show left
            iiiiii = 1
            x_plots = 3
            y_plots = 2
            axes_left_list = list()
            axes_right_list = list()

            im_left = fix_rgb_image(left_blob[0])
            ax1 = fig.add_subplot(y_plots, x_plots, iiiiii)
            ax1.imshow(im_left)
            ax1.set_title("left image")
            iiiiii += 1
            axes_left_list.append(ax1)


            # show right
            im_right = fix_rgb_image(right_blob[0])
            ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
            ax2.imshow(im_right)
            ax2.set_title("right image (red dot is predicted flow, green is ground truth, and blue is the location on the right image)")
            iiiiii += 1
            axes_right_list.append(ax2)

            # create flow images, but don't display them yet
            gt_flow_color_square = sintel_utils.sintel_compute_color(gt_flow)
            gt_flow_raw_color = sintel_utils.raw_color_from_flow(gt_flow)
            gt_flow_plot_position = iiiiii
            iiiiii += 1

            computed_flows_plot_positions = list()
            computed_flows_color_square = list()
            computed_flows_raw_color = list()
            for i in range(len(flow_arrays)):
                computed_flows_color_square.append(sintel_utils.sintel_compute_color(flow_arrays[i]))
                computed_flows_raw_color.append(sintel_utils.raw_color_from_flow(flow_arrays[i]))
                computed_flows_plot_positions.append(iiiiii)
                iiiiii += 1

            gt_flow_ax = None
            def plot_flow_images(color_square_not_raw):
                gt_flow_ax = fig.add_subplot(y_plots, x_plots, gt_flow_plot_position)
                # ax3.imshow(gt_flow_im)
                if color_square_not_raw:
                    gt_flow_ax.imshow(gt_flow_color_square)
                else:
                    gt_flow_ax.imshow(gt_flow_raw_color)
                gt_flow_ax.set_title("gt flow")
                axes_left_list.append(gt_flow_ax)

                for i in range(len(flow_arrays)):
                    ax7 = fig.add_subplot(y_plots, x_plots, computed_flows_plot_positions[i])
                    if color_square_not_raw:
                        ax7.imshow(computed_flows_color_square[i])
                    else:
                        ax7.imshow(computed_flows_raw_color[i])
                    ax7.set_title("raw predicted flow at scale " + str(i))
                    axes_left_list.append(ax7)

            plot_flow_images(False)

            ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
            ax2.imshow(np.squeeze(depth_blob[0]))
            ax2.set_title("depth")
            iiiiii += 1
            axes_right_list.append(ax2)


            # l_features = np.squeeze(results[0])
            # r_features = np.squeeze(results[1])
            # # warp left image
            # # warped = np.zeros([l_features.shape[0] / 4, l_features.shape[1] / 4, l_features.shape[2]])
            # # # warped = np.copy(image_left)
            # # # warped = np.copy(image_right)
            # # for i in range(0, l_features.shape[0], 4):
            # #     for j in range(0, l_features.shape[1], 4):
            # #             i_new = i + int(gt_flow[i, j, 1])
            # #             j_new = j + int(gt_flow[i, j, 0])
            # #             if 0 <= i_new < warped.shape[0] and 0 <= j_new < warped.shape[1]:
            # #                 warped[i/4, j/4] = l_features[i_new, j_new] - r_features[i, j]
            # #
            # feature_scale_min, feature_scale_max = sintel_utils.colorize_features(l_features, get_scale=True)
            # # ax4 = fig.add_subplot(334)
            # # ax4.imshow(sintel_utils.colorize_features(warped, scale_low=feature_scale_min, scale_high=feature_scale_max))
            # # ax4.set_title("sparse feature difference")
            #
            # # show flow differences
            # gt_components = np.split(gt_flow, 2, axis=2)
            # pred_components = np.split(predicted_flow_cropped, 2, axis=2)
            # gt_angle = np.arctan2(gt_components[1], gt_components[0])
            # pred_angle = np.arctan2(pred_components[1], pred_components[0])
            # gt_mag = np.sqrt(np.power(gt_components[0], 2)+ np.power(gt_components[1], 2))
            # pred_mag = np.sqrt(np.power(pred_components[0], 2) + np.power(pred_components[1], 2))
            #
            # angle_dif = np.mod((gt_angle - pred_angle) + np.pi, np.pi * 2) - np.pi
            #
            # ax5 = fig.add_subplot(y_plots, x_plots, iiiiii)
            # ax5.imshow(np.abs(angle_dif.squeeze()) / np.pi * (np.abs(gt_mag - pred_mag).squeeze() / np.max(gt_mag)), cmap='Greys')
            # ax5.set_title("direction difference * magnitude difference")
            # ax5.set_xlabel("white = no error, black = large error")
            # iiiiii += 1
            # axes_left_list.append(ax5)
            #
            # ax6 = fig.add_subplot(y_plots, x_plots, iiiiii)
            # ax6.imshow(np.abs(gt_mag - pred_mag).squeeze() / np.max(gt_mag), cmap='Greys')
            # ax6.set_title("magnitude difference")
            # ax6.set_xlabel("white = no error, black = large error")
            # iiiiii += 1
            # axes_left_list.append(ax6)
            #
            #
            # similar_r = utils.plotting_tools.feature_similarity(l_features, r_features, gt_flow)
            # # similar_r = np.zeros(l_features.shape[0:2])
            # # # warped = np.copy(image_left)
            # # # warped = np.copy(image_right)
            # # for i in range(0, similar_r.shape[0]):
            # #     for j in range(0, similar_r.shape[1]):
            # #         i_new = i + int(gt_flow[i, j, 1])
            # #         j_new = j + int(gt_flow[i, j, 0])
            # #         if 0 <= i_new < similar_r.shape[0]:
            # #                 if 0 <= j_new < similar_r.shape[1]:
            # #                     similar_r[i, j] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))
            #
            #
            # similar_r_pred = utils.plotting_tools.feature_similarity(l_features, r_features, predicted_flow)
            # # similar_r_pred = np.zeros(l_features.shape[0:2])
            # # # warped = np.copy(image_left)
            # # # warped = np.copy(image_right)
            # # for i in range(0, similar_r_pred.shape[0], 2):
            # #     for j in range(0, similar_r_pred.shape[1], 2):
            # #         i_new = i + int(predicted_flow[i, j, 1])
            # #         j_new = j + int(predicted_flow[i, j, 0])
            # #         if 0 <= i_new < similar_r_pred.shape[0]:
            # #                 if 0 <= j_new < similar_r.shape[1]:
            # #                     similar_r_pred[i, j] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))
            # #                     similar_r_pred[i+1, j] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))
            # #                     similar_r_pred[i, j+1] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))
            # #                     similar_r_pred[i+1, j+1] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))
            # # similar_r = np.dstack([similar_r + np.squeeze(occluded_blob[0]), similar_r, similar_r])
            #
            # # similar = np.zeros(l_features.shape[0:2])
            # # # warped = np.copy(image_left)
            # # # warped = np.copy(image_right)
            # # box_width = 14
            # # for i in range(0, similar.shape[0], box_width):
            # #     for j in range(0, similar.shape[1], box_width):
            # #         for a in range(box_width / -2 + 1, box_width / 2, 2):
            # #             if 0 <= i + a < similar.shape[0]:
            # #                 for b in range(box_width / -2 + 1, box_width / 2, 2):
            # #                     if 0 <= j + b < similar.shape[1]:
            # #                         similar[i + a, j + b] = np.sqrt(
            # #                             np.sum(np.power(l_features[i, j] - l_features[i + a, j + b], 2)))
            # #                         similar[i+a+1, j+b] = similar[i + a, j + b]
            # #                         similar[i+a, j+b+1] = similar[i + a, j + b]
            # #                         similar[i+a+1, j+b+1] = similar[i + a, j + b]
            # #         similar[i, j] = 1
            #
            # # ax5 = fig.add_subplot(y_plots, x_plots, iiiiii)
            # # ax5.imshow(similar, cmap='Greys')
            # # ax5.set_title("pixel neighbor similarity")
            # # ax5.set_xlabel("white = no error, black = large error")
            # # iiiiii += 1
            # # axes_left_list.append(ax5)
            #
            # ax5_ = fig.add_subplot(y_plots, x_plots, iiiiii)
            # # ax5_.imshow(np.squeeze(occluded_blob[0]), cmap='Reds')
            # ax5_.imshow(similar_r_pred, cmap='Greys')
            # ax5_.set_title("corresponding feature similarity (predicted flow)")
            # ax5_.set_xlabel("white = no error, black = large error, red means occluded")
            # iiiiii += 1
            # axes_left_list.append(ax5_)
            #
            # ax5_ = fig.add_subplot(y_plots, x_plots, iiiiii)
            # # ax5_.imshow(np.squeeze(occluded_blob[0]), cmap='Reds')
            # ax5_.imshow(similar_r, cmap='Greys')
            # ax5_.set_title("corresponding feature similarity (gt flow)")
            # ax5_.set_xlabel("white = no error, black = large error, red means occluded")
            # iiiiii += 1
            # axes_left_list.append(ax5_)
            #
            #
            # ax_l_features = fig.add_subplot(y_plots, x_plots, iiiiii)
            # ax_l_features.imshow(sintel_utils.colorize_features(np.squeeze(results[0]), scale_low=feature_scale_min, scale_high=feature_scale_max))
            # ax_l_features.set_title("left features")
            # iiiiii += 1
            # axes_left_list.append(ax_l_features)
            #
            # # ax_r_features = fig.add_subplot(y_plots, x_plots, iiiiii)
            # # ax_r_features.imshow(sintel_utils.colorize_features(np.squeeze(results[1]), scale_low=feature_scale_min, scale_high=feature_scale_max))
            # # ax_r_features.set_title("right features")
            # # iiiiii += 1
            # # axes_right_list.append(ax_r_features)

            # fig.suptitle('Image ' + str(images['image_left']) + '\naverage endpoint error: ' + str(average_EPE) +
            #              ' (predicting no movement would result in EPE of ' + str(zero_prediction_EPE) + ')' +
            #              "\ncalculated triplet loss is %7.4f" % float(results[3][0]), fontsize=10)

            x_points = list()
            y_points = list()
            x_points_right = list()
            y_points_right = list()
            colors = list()

            global red_point, green_point, blue_point
            red_point = None
            green_point = None
            blue_point = None

            # def onclick(event):
            #     print(
            #     'button', event.button, 'x=', event.x, 'y=', event.y, 'xdata=', event.xdata, 'ydata=', event.ydata)
            #     for ax in axes_left_list:
            #         data_transformer = ax.transData.inverted()
            #         x_left, y_left = data_transformer.transform([event.x, event.y])
            #         if -1 <= x_left <= l_features.shape[1] + 3 and -1 <= y_left <= l_features.shape[0] + 3:
            #             print("\t x transformed:", x_left, "y transformed:", y_left)
            #             if event.xdata is not None and event.ydata is not None:
            #                 x_points_right.append(event.xdata + int(predicted_flow[int(event.ydata), int(event.xdata), 0]))
            #                 y_points_right.append(event.ydata + int(predicted_flow[int(event.ydata), int(event.xdata), 1]))
            #                 x_points.append(event.xdata)
            #                 y_points.append(event.ydata)
            #                 colors.append(random.random())
            #
            #                 for sub_ax in axes_left_list:
            #                     sub_ax.scatter(x_points, y_points, c=colors, s=5, marker='p')
            #                 for sub_ax in axes_right_list:
            #                     global red_point, green_point, blue_point
            #
            #                     if red_point is not None:
            #                         blue_point.remove()
            #                         red_point.remove()
            #                         green_point.remove()
            #                     blue_point = sub_ax.scatter([event.xdata], [event.ydata], c='BLUE', s=5, marker='p')
            #                     red_point = sub_ax.scatter([event.xdata + int(predicted_flow[int(event.ydata), int(event.xdata), 0])],
            #                                    [event.ydata + int(predicted_flow[int(event.ydata), int(event.xdata), 1])], c='RED', s=5, marker='p')
            #                     green_point = sub_ax.scatter([event.xdata + int(gt_flow[int(event.ydata), int(event.xdata), 0])],
            #                                    [event.ydata + int(gt_flow[int(event.ydata), int(event.xdata), 1])], c='GREEN', s=5, marker='p')
            #                 fig.canvas.draw()
            #             break
            #
            # cid = fig.canvas.mpl_connect('button_press_event', onclick)

            def handle_key_press(event):
                if event.key == 'c':
                    plot_flow_images(True)
                elif event.key == 'r':
                    plot_flow_images(False)
                else:
                    print "key not tied to any action"
                # only redraw if something changed
                fig.canvas.draw()

            fig.canvas.mpl_connect('key_press_event', handle_key_press)

            plt.interactive(False)
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.11)
            if save_image:
                plt.savefig("plot_" + str(training_iter) + "_" + str(i) + ".png")
            else:
                plt.show()
            plt.close('all')
    if calculate_EPE_all_data:
        average = np.mean(EPE_list)
        print "# average EPE is " + str(average) + " for entire " + str(imdb._name) + " dataset with network " + \
            str(weights_filename)


def fix_rgb_image(image_in):
    image = image_in.copy() + cfg.PIXEL_MEANS
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    return image


def calculate_flow_single_frame(sess, net, im_left, im_right):
    # compute image blob
    left_blob, right_blob, im_scales = _get_flow_image_blob(im_left, im_right, 0)

    feed_dict = {net.data_left: left_blob, net.data_right: right_blob,
                 net.gt_flow: np.zeros([left_blob.shape[0], left_blob.shape[1], left_blob.shape[2], 2],
                                       dtype=np.float32), net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    output_flow = sess.run([net.get_output('predicted_flow')])
    return output_flow


def siphon_flow_single_frame(sess, net, im_left, im_right):
    # compute image blob
    left_blob, right_blob, im_scales = _get_flow_image_blob(im_left, im_right, 0)

    training_data_queue = list()
    queue_start_size = sess.run(net.queue_size_op)
    while sess.run(net.queue_size_op) != 0:
        training_data_queue.append(sess.run({'left':net.get_output('data_left'), 'right':net.get_output('data_right'),
                                             'flow':net.get_output('gt_flow'), 'keep_prob':net.keep_prob_queue}))

    feed_dict = {net.data_left: left_blob, net.data_right: right_blob,
                 net.gt_flow: np.zeros([left_blob.shape[0], left_blob.shape[1], left_blob.shape[2], 2],
                                       dtype=np.float32), net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    output = sess.run({'flow':net.get_output('predicted_flow'), 'left':net.get_output('data_left_tap'),
                            'right':net.get_output('data_left_tap')})

    for i in training_data_queue:
        feed_dict = {net.data_left: i['left'], net.data_right: i['right'], net.gt_flow: i['flow'], net.keep_prob: i['keep_prob']}
        sess.run(net.enqueue_op, feed_dict=feed_dict)

    # assert sess.run(net.queue_size_op) == queue_start_size, "data queue size changed"
    return output


def siphon_outputs_single_frame(sess, net, data_feed_dict, outputs):
    # compute image blob

    training_data_queue = list()
    queue_start_size = sess.run(net.queue_size_op)
    while sess.run(net.queue_size_op) != 0:
        training_data_queue.append(sess.run({'left':net.get_output('data_left'), 'right':net.get_output('data_right'),
                                             'flow':net.get_output('gt_flow'), 'keep_prob':net.keep_prob_queue}))

    sess.run(net.enqueue_op, feed_dict=data_feed_dict)
    output = sess.run(outputs)

    for i in training_data_queue:
        feed_dict = {net.data_left: i['left'], net.data_right: i['right'], net.gt_flow: i['flow'], net.keep_prob: i['keep_prob']}
        sess.run(net.enqueue_op, feed_dict=feed_dict)

    # assert sess.run(net.queue_size_op) == queue_start_size, "data queue size changed"
    return output


def _get_flow_image_blob(im_left, im_right, scale_ind):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    num_images = 1
    processed_left = []
    processed_right = []
    processed_flow = []
    im_scales = []

    # left image
    im_left = pad_im(cv2.imread(im_left, cv2.IMREAD_UNCHANGED), 16)
    if im_left.shape[2] == 4:
        im = np.copy(im_left[:, :, :3])
        alpha = im_left[:, :, 3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 0
        im_lef = im

    im_right = pad_im(cv2.imread(im_right, cv2.IMREAD_UNCHANGED), 16)
    if im_left.shape[2] == 4:
        im = np.copy(im_left[:, :, :3])
        alpha = im_right[:, :, 3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 0
        im_right = im


    # TODO: is this important?
    im_scale = cfg.TEST.SCALES_BASE[scale_ind]
    im_scales.append(im_scale)

    im_left_orig = im_left.astype(np.float32, copy=True)
    im_left_orig -= cfg.PIXEL_MEANS
    im_left_processed = cv2.resize(im_left_orig, None, None, fx=im_scale, fy=im_scale,
                                   interpolation=cv2.INTER_LINEAR)
    processed_left.append(im_left_processed)

    im_right_orig = im_right.astype(np.float32, copy=True)
    im_right_orig -= cfg.PIXEL_MEANS
    im_right_processed = cv2.resize(im_right_orig, None, None, fx=im_scale, fy=im_scale,
                                    interpolation=cv2.INTER_LINEAR)
    processed_right.append(im_right_processed)


    # Create a blob to hold the input images
    image_left_blob = im_list_to_blob(processed_left, 3)
    image_right_blob = im_list_to_blob(processed_right, 3)
    blob_rescale = []

    return image_left_blob, image_right_blob, im_scales


##################
# test video
##################
def test_net(sess, net, imdb, weights_filename, rig_filename, is_kfusion):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    print imdb.name
    if os.path.exists(seg_file):
        with open(seg_file, 'rb') as fid:
            segmentations = cPickle.load(fid)
        imdb.evaluate_segmentations(segmentations, output_dir)
        return

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)
    voxelizer.setup(-3, -3, -3, 3, 3, 4)
    # voxelizer.setup(-2, -2, -2, 2, 2, 2)

    # kinect fusion
    if is_kfusion:
        KF = kfusion.PyKinectFusion(rig_filename)

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
    else:
        perm = xrange(num_images)

    video_index = ''
    have_prediction = False
    for i in perm:
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        height = rgba.shape[0]
        width = rgba.shape[1]

        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            have_prediction = False
            state = np.zeros((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            weights = np.ones((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            points = np.zeros((1, height, width, 3), dtype=np.float32)
        else:
            if video_index != image_index[:pos]:
                have_prediction = False
                video_index = image_index[:pos]
                state = np.zeros((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                weights = np.ones((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                points = np.zeros((1, height, width, 3), dtype=np.float32)
                print 'start video {}'.format(video_index)

        # read color image
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # read depth image
        im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        # backprojection for the first frame
        if not have_prediction:    
            if is_kfusion:
                # KF.set_voxel_grid(-3, -3, -3, 6, 6, 7)
                KF.set_voxel_grid(voxelizer.min_x, voxelizer.min_y, voxelizer.min_z, voxelizer.max_x-voxelizer.min_x, voxelizer.max_y-voxelizer.min_y, voxelizer.max_z-voxelizer.min_z)
                # identity transformation
                RT_world = np.zeros((3,4), dtype=np.float32)
                RT_world[0, 0] = 1
                RT_world[1, 1] = 1
                RT_world[2, 2] = 1
            else:
                # store the RT for the first frame
                RT_world = meta_data['rotation_translation_matrix']

        # run kinect fusion
        if is_kfusion:
            im_rgb = np.copy(im)
            im_rgb[:, :, 0] = im[:, :, 2]
            im_rgb[:, :, 2] = im[:, :, 0]
            KF.feed_data(im_depth, im_rgb, im.shape[1], im.shape[0], float(meta_data['factor_depth']))
            KF.back_project();
            if have_prediction:
                pose_world2live, pose_live2world = KF.solve_pose()
                RT_live = pose_world2live
            else:
                RT_live = RT_world
        else:
            # compute camera poses
            RT_live = meta_data['rotation_translation_matrix']

        pose_world2live = se3_mul(RT_live, se3_inverse(RT_world))
        pose_live2world = se3_inverse(pose_world2live)

        _t['im_segment'].tic()
        labels, probs, state, weights, points = im_segment(sess, net, im, im_depth, state, weights, points, meta_data, voxelizer, pose_world2live, pose_live2world)
        _t['im_segment'].toc()
        # time.sleep(3)

        _t['misc'].tic()
        labels = unpad_im(labels, 16)

        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        if is_kfusion:
            labels_kfusion = np.zeros((height, width), dtype=np.int32)
            if probs.shape[2] < 10:
                probs_new = np.zeros((probs.shape[0], probs.shape[1], 10), dtype=np.float32)
                probs_new[:,:,:imdb.num_classes] = probs
                probs = probs_new
            KF.feed_label(im_label, probs, colors)
            KF.fuse_depth()
            labels_kfusion = KF.extract_surface(labels_kfusion)
            im_label_kfusion = imdb.labels_to_image(im, labels_kfusion)
            KF.render()
            filename = os.path.join(output_dir, 'images', '{:04d}'.format(i))
            KF.draw(filename, 1)
        have_prediction = True

        # compute the delta transformation between frames
        RT_world = RT_live

        if is_kfusion:
            seg = {'labels': labels_kfusion}
        else:
            seg = {'labels': labels}
        segmentations[i] = seg

        _t['misc'].toc()

        if cfg.TEST.VISUALIZE:
            # read label image
            labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)
            if len(labels_gt.shape) == 2:
                im_label_gt = imdb.labels_to_image(im, labels_gt)
            else:
                im_label_gt = np.copy(labels_gt[:,:,:3])
                im_label_gt[:,:,0] = labels_gt[:,:,2]
                im_label_gt[:,:,2] = labels_gt[:,:,0]
            vis_segmentations(im, im_depth, im_label, im_label_gt, imdb._class_colors)

        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

    if is_kfusion:
        KF.draw(filename, 1)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)

# compute the voting label image in 2D
def _vote_centers(im_label, cls_indexes, centers, num_classes):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)

    center = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        if len(x) > 0:
            ind = np.where(cls_indexes == i)[0]
            center[0] = centers[ind, 0]
            center[1] = centers[ind, 1]
            R = np.tile(center, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            vertex_targets[y, x, 0] = R[0,:]
            vertex_targets[y, x, 1] = R[1,:]

    return vertex_targets


# extract vertmap for vertex predication
def _extract_vertmap(im_label, vertex_pred, extents, num_classes):
    height = im_label.shape[0]
    width = im_label.shape[1]
    vertmap = np.zeros((height, width, 2), dtype=np.float32)
    # centermap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(im_label == i)
        if len(I[0]) > 0:
            start = 2 * i
            end = 2 * i + 2
            vertmap[I[0], I[1], :] = vertex_pred[0, I[0], I[1], start:end]

            # start = 2 * i
            # end = 2 * i + 2
            # centermap[I[0], I[1], :2] = vertex_pred[0, I[0], I[1], start:end]

    return vertmap
    #return _unscale_vertmap(vertmap, im_label, extents, num_classes)
    #return vertmap, centermap  


def scale_vertmap(vertmap):
    vmin = vertmap.min()
    vmax = vertmap.max()
    if vmax - vmin > 0:
        a = 1.0 / (vmax - vmin)
        b = -1.0 * vmin / (vmax - vmin)
    else:
        a = 0
        b = 0
    return a * vertmap + b


def _unscale_vertmap(vertmap, labels, extents, num_classes):
    for k in range(1, num_classes):
        index = np.where(labels == k)
        for i in range(3):
            vmin = -extents[k, i] / 2
            vmax = extents[k, i] / 2
            a = 1.0 / (vmax - vmin)
            b = -1.0 * vmin / (vmax - vmin)
            vertmap[index[0], index[1], i] = (vertmap[index[0], index[1], i] - b) / a
    return vertmap


def vis_segmentations_vertmaps(im, im_depth, im_labels, im_labels_gt, colors, vertmap_gt, vertmap, labels, labels_gt, centers, intrinsic_matrix):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(241)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')
    ax.autoscale()

    # show gt class labels
    ax = fig.add_subplot(242)
    plt.imshow(im_labels_gt)
    ax.set_title('gt class labels')
    ax.autoscale()

    # show depth
    ax = fig.add_subplot(245)
    plt.imshow(im_depth)
    ax.set_title('input depth')
    ax.autoscale()

    # show gt vertex map
    ax = fig.add_subplot(243)
    plt.imshow(vertmap_gt[:,:,0])
    ax.set_title('gt centers x')
    ax.autoscale()

    ax = fig.add_subplot(244)
    plt.imshow(vertmap_gt[:,:,1])
    ax.set_title('gt centers y')
    ax.autoscale()

    # show class label
    ax = fig.add_subplot(246)
    plt.imshow(im_labels)
    ax.set_title('class labels')
    ax.autoscale()

    # show centers
    index = np.where(np.isfinite(centers[:, 0]))[0]
    plt.plot(centers[index, 0], centers[index, 1], 'ro')

    # show boxes
    for i in xrange(len(index)):
        roi = centers[index[i], :]
        plt.gca().add_patch(
            plt.Rectangle((roi[0] - roi[2]/2, roi[1] - roi[3]/2), roi[2],
                          roi[3], fill=False,
                          edgecolor='g', linewidth=3)
            )

    # show vertex map
    ax = fig.add_subplot(247)
    plt.imshow(vertmap[:,:,0])
    ax.set_title('centers x')
    ax.autoscale()

    ax = fig.add_subplot(248)
    plt.imshow(vertmap[:,:,1])
    ax.set_title('centers y')
    ax.autoscale()

    # show projection of the poses
    if cfg.TEST.RANSAC:
        ax = fig.add_subplot(234, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        num_classes = poses.shape[2]
        for i in xrange(1, num_classes):
            index = np.where(labels_gt == i)
            if len(index[0]) > 0:
                if np.isinf(poses[0, 0, i]):
                    print 'missed object {}'.format(i)
                else:
                    # projection
                    RT = poses[:, :, i]
                    print RT

                    num = len(index[0])
                    # extract 3D points
                    x3d = np.ones((4, num), dtype=np.float32)
                    x3d[0, :] = vertmap_gt[index[0], index[1], 0] / cfg.TRAIN.VERTEX_W
                    x3d[1, :] = vertmap_gt[index[0], index[1], 1] / cfg.TRAIN.VERTEX_W
                    x3d[2, :] = vertmap_gt[index[0], index[1], 2] / cfg.TRAIN.VERTEX_W

                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[i], 255.0), alpha=0.05)
        ax.set_title('projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

    plt.tight_layout()
    plt.show()


###################
# test single frame
###################
def test_net_single_frame(sess, net, imdb, weights_filename, rig_filename, is_kfusion):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    print imdb.name
    if os.path.exists(seg_file):
        with open(seg_file, 'rb') as fid:
            segmentations = cPickle.load(fid)
        imdb.evaluate_segmentations(segmentations, output_dir)
        return

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # kinect fusion
    if is_kfusion:
        KF = kfusion.PyKinectFusion(rig_filename)

    # pose estimation
    if cfg.TEST.VERTEX_REG:
        RANSAC = ransac.PyRansac3D()

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
        # perm = xrange(0, num_images, 5)
    else:
        perm = xrange(num_images)

    video_index = ''
    have_prediction = False
    for i in perm:

        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            have_prediction = False
        else:
            if video_index != image_index[:pos]:
                have_prediction = False
                video_index = image_index[:pos]
                print 'start video {}'.format(video_index)

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # read depth image
        im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        # read label image
        labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if len(labels_gt.shape) == 2:
            im_label_gt = imdb.labels_to_image(im, labels_gt)
        else:
            im_label_gt = np.copy(labels_gt[:,:,:3])
            im_label_gt[:,:,0] = labels_gt[:,:,2]
            im_label_gt[:,:,2] = labels_gt[:,:,0]

        _t['im_segment'].tic()
        labels, probs, vertex_pred = im_segment_single_frame(sess, net, im, im_depth, meta_data, imdb.num_classes)
        if cfg.TEST.VERTEX_REG:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            centers = RANSAC.estimate_center(probs, vertex_pred[0,:,:,:])
            print centers
            if cfg.TEST.RANSAC:
                # pose estimation using RANSAC
                fx = meta_data['intrinsic_matrix'][0, 0]
                fy = meta_data['intrinsic_matrix'][1, 1]
                px = meta_data['intrinsic_matrix'][0, 2]
                py = meta_data['intrinsic_matrix'][1, 2]
                depth_factor = meta_data['factor_depth'][0, 0]
                poses = RANSAC.estimate_pose(im_depth, probs, vertex_pred[0,:,:,:] / cfg.TRAIN.VERTEX_W, imdb._extents, fx, fy, px, py, depth_factor)

                # print gt poses
                # cls_indexes = meta_data['cls_indexes']
                # poses_gt = meta_data['poses']
                # for j in xrange(len(cls_indexes)):
                #    print 'object {}'.format(cls_indexes[j])
                #    print poses_gt[:,:,j]
            else:
                poses = []

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels = unpad_im(labels, 16)
        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        if not have_prediction:    
            if is_kfusion:
                KF.set_voxel_grid(-3, -3, -3, 6, 6, 7)

        # run kinect fusion
        if is_kfusion:
            height = im.shape[0]
            width = im.shape[1]
            labels_kfusion = np.zeros((height, width), dtype=np.int32)

            im_rgb = np.copy(im)
            im_rgb[:, :, 0] = im[:, :, 2]
            im_rgb[:, :, 2] = im[:, :, 0]
            KF.feed_data(im_depth, im_rgb, im.shape[1], im.shape[0], float(meta_data['factor_depth']))
            KF.back_project();
            if have_prediction:
                pose_world2live, pose_live2world = KF.solve_pose()

            KF.feed_label(im_label, probs, colors)
            KF.fuse_depth()
            labels_kfusion = KF.extract_surface(labels_kfusion)
            im_label_kfusion = imdb.labels_to_image(im, labels_kfusion)
            KF.render()
            filename = os.path.join(output_dir, 'images', '{:04d}'.format(i))
            KF.draw(filename, 0)
        have_prediction = True

        if is_kfusion:
            seg = {'labels': labels_kfusion}
        else:
            seg = {'labels': labels}
        segmentations[i] = seg

        _t['misc'].toc()

        print 'im_segment {}: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(video_index, i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

        if cfg.TEST.VISUALIZE:
            if cfg.TEST.VERTEX_REG:
                centers_gt = _vote_centers(labels_gt, meta_data['cls_indexes'], meta_data['center'], imdb.num_classes)
                print 'visualization'
                vis_segmentations_vertmaps(im, im_depth, im_label, im_label_gt, imdb._class_colors, \
                    centers_gt, vertmap, labels, labels_gt, centers, meta_data['intrinsic_matrix'])
            else:
                vis_segmentations(im, im_depth, im_label, im_label_gt, imdb._class_colors)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)



###################
# test GAN
###################

def vis_gan(im, im_depth, vertmap, vertmap_gt):

    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(221)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(222)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(223)
    plt.imshow(vertmap)
    ax.set_title('vertmap')

    ax = fig.add_subplot(224)
    plt.imshow(vertmap_gt)
    ax.set_title('gt vertmap')

    plt.show()


def test_gan(sess, net, imdb, weights_filename):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """Test a GAN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
    else:
        perm = xrange(num_images)

    for i in perm:

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # read depth image
        im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        _t['im_segment'].tic()

        im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

        height = im.shape[0]
        width = im.shape[1]
        vertex_image_blob = np.zeros((1, height, width, 3), dtype=np.float32)
        vertmap = pad_im(cv2.imread(imdb.vertmap_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        vertex_image_blob[0, :, :, :] = vertmap.astype(np.float32) / 127.5 - 1

        gan_z_blob = np.random.uniform(-1, 1, [1, 100]).astype(np.float32)

        feed_dict = {net.data: im_rescale_blob, net.data_gt: vertex_image_blob, net.z: gan_z_blob, net.keep_prob: 1.0}

        sess.run(net.enqueue_op, feed_dict=feed_dict)
        output_g = sess.run([net.get_output('output_g')], feed_dict=feed_dict)
        labels = output_g[0][0,:,:,:]
        labels = (labels + 1) * 127.5
        print labels.shape

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels = unpad_im(labels, 16)

        seg = {'labels': labels}
        segmentations[i] = seg

        _t['misc'].toc()

        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

        if cfg.TEST.VISUALIZE:
            vis_gan(im, im_depth, labels, vertmap)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

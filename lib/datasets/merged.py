__author__ = 'ali & kwon'

import os
import datasets
import datasets.merged
import datasets.imdb
import cPickle
import numpy as np
import cv2

class merged(datasets.imdb):
    def __init__(self, image_set, merged_path = None):#image_set='train' or 'val'
        datasets.imdb.__init__(self, 'merged_' + image_set)
        self._image_set = image_set
        self._merged_path = self._get_default_path() if merged_path is None \
                            else merged_path
        self._data_path = os.path.join(self._merged_path, 'data')
        self._classes = ('bowl', 'box', 'chair', 'couch', 'desk', 'hat', 'cup', 'table', 'can', 'monitor', 'bottle', 'keyboard', '__background__')
        self._class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),(128,128,0),(128,0,128),(0,0,0)]
        self._class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1]
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._merged_path), \
                'merged path does not exist: {}'.format(self._merged_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        if(index.find('scene') >= 0):
            image_path = os.path.join(self._data_path, index + '-color' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path
        else:
            image_path = os.path.join(self._data_path, index + '_rgba' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path


    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        if(index.find('scene') >= 0):
            image_path = os.path.join(self._data_path, index + '-depth' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path
        else:
            image_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        if(index.find('scene') >= 0):
            image_path = os.path.join(self._data_path, index + '-label' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            print(image_path)
            return image_path
        else:
            image_path = os.path.join(self._data_path, index + '_label' + self._image_ext)
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            print(image_path)
            return image_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        if(index.find('scene') >= 0):
            image_path = os.path.join(self._data_path, index + '-meta.mat')
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path
        else:
            image_path = os.path.join(self._data_path, index + '_meta.mat')
            assert os.path.exists(image_path), \
                  'Path does not exist: {}'.format(image_path)
            return image_path


    #yolo path
    def yolo_path_at(self, i):

        return self.yolo_path_from_index(self.image_index[i])

    def yolo_path_from_index(self, index):
        label_path = os.path.join(self._data_path, index + '-yolo.txt')
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    def scene_label_at(self, i):
        return self.scene_label_from_index(self.image_index[i])

    def scene_label_from_index(self, index):
        if(index.find('scene') >= 0):
            return 1
        else:
            return 0

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._merged_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'MergedDataset')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_merged_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_merged_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)
        print(label_path,"we are hereeeeee")
        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # YOLO path
        yolo_path = self.yolo_path_from_index(index)

        # scene label
        scene_label = self.scene_label_from_index(index)

        # parse image name
        pos = index.find('/')
        video_id = index[:pos]

        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'video_id': video_id,
                'class_colors': self._class_colors,
                'class_weights': self._class_weights,
                'flipped': False,
                'yolo': yolo_path,
                'scene_label': scene_label}

    def _process_label_image(self, label_image):
        """
        change label image to label index
        """
        class_colors = self._class_colors
        width = label_image.shape[1]
        height = label_image.shape[0]
        label_index = np.zeros((height, width), dtype=np.float32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I] = i

        return label_index


    def labels_to_image(self, im, labels):
        class_colors = self._class_colors
        height = labels.shape[0]
        width = labels.shape[1]
        image_r = np.zeros((height, width), dtype=np.float32)
        image_g = np.zeros((height, width), dtype=np.float32)
        image_b = np.zeros((height, width), dtype=np.float32)

        for i in xrange(len(class_colors)):
            color = class_colors[i]
            I = np.where(labels == i)
            image_r[I] = color[0]
            image_g[I] = color[1]
            image_b[I] = color[2]

        image = np.stack((image_r, image_g, image_b), axis=-1)
        # index = np.where(image == 255)
        # image[index] = im[index]
        # image = 0.1*im + 0.9*image

        return image.astype(np.uint8)

    def _process_label_image(self, label_image):
        """
        change label image to label index
        """
        class_colors = self._class_colors
        width = label_image.shape[1]
        height = label_image.shape[0]
        label_index = np.zeros((height, width), dtype=np.float32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I] = i

        return label_index

    def evaluate_segmentations(self, segmentations, output_dir):
        print 'evaluating segmentations'
        # compute histogram
        n_cl = self.num_classes
        hist = np.zeros((n_cl, n_cl))
        print(n_cl,"   number of classes")
        # make image dir
        image_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # for each image
        for im_ind, index in enumerate(self.image_index):
            print(im_ind, index)

            # read ground truth labels
            im = cv2.imread(self.label_path_from_index(index), cv2.IMREAD_UNCHANGED)
            #if (im.shape)
            #gt_labels = im.astype(np.float32)
            #if(len(im.shape) == 3):
            #    gt_labels = self._process_label_image(im)
            #else:
            gt_labels = im.astype(np.float32)
            # predicated labels
            sg_labels = segmentations[im_ind]['labels']
            cv2.imwrite(image_dir+"/justlabel"+str(im_ind)+'.png',gt_labels)
            #print(sg_labels.shape," sggggggggggggggggggggg")
            hist += self.fast_hist(gt_labels.flatten(), sg_labels.flatten(), n_cl)


        # overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        print 'overall accuracy', acc
        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print 'mean accuracy', np.nanmean(acc)
        # per-class IU
        print 'per-class IU'
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        for i in range(n_cl):
            print '{} {}'.format(self._classes[i], iu[i])
        print 'mean IU', np.nanmean(iu)
        freq = hist.sum(1) / hist.sum()
        print 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()

        filename = os.path.join(output_dir, 'segmentation.txt')
        with open(filename, 'wt') as f:
            for i in range(n_cl):
                f.write('{:f}\n'.format(iu[i]))
        filename2 = os.path.join(output_dir, 'conf_segmentation.txt')
        with open(filename2, 'wt') as f2:
            for i in range(n_cl):
                for j in range(n_cl):
                    f2.write('{:f}  '.format(hist[i][j]/hist.sum(1)[i]))
                f2.write("\n")

        filename3 = os.path.join(output_dir, 'conf_unnorm_segmentation.txt')
        with open(filename3, 'wt') as f3:
            for i in range(n_cl):
                for j in range(n_cl):
                    f3.write('{:f}  '.format(hist[i][j]))
                f3.write("\n")

if __name__ == '__main__':
    d = datasets.merged('train')
    res = d.roidb
    from IPython import embed; embed()

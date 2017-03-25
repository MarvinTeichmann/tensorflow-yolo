from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
import numpy as np
from Queue import Queue
from threading import Thread

from yolo.dataset.dataset import DataSet

import scipy as scp
import scipy.misc


class TextDataSet(DataSet):

    """TextDataSet
    process text input file dataset
    text file format:
      image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
    """

    def __init__(self, common_params, dataset_params):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        # process params
        self.base_path = str(dataset_params['data_dir'])
        self.train_txt = str(dataset_params['train_txt'])
        self.val_txt = str(dataset_params['val_txt'])
        self.width = int(common_params['image_width'])
        self.height = int(common_params['image_height'])
        self.batch_size = int(common_params['batch_size'])
        self.thread_num = int(dataset_params['thread_num'])
        self.max_objects = int(common_params['max_objects_per_image'])

        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)

        self.record_list = []

        self.train_txt = os.path.join(dataset_params['data_dir'],
                                      dataset_params['train_txt'])

        # filling the record_list
        input_file = open(self.train_txt, 'r')

        for line in input_file:
            line = line.strip()
            self.record_list.append(line)

        self.record_point = 0
        self.record_number = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """record_queue's processor
        """
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def read_kitti_anno(self, label_file, orig_shape, detect_truck):
        """ Reads a kitti annotation file.

        Args:
        label_file: Path to file

        Returns:
          Lists of rectangels: Cars and don't care area.
        """
        h = orig_shape[0]
        w = orig_shape[1]

        width_rate = self.width * 1.0 / w
        height_rate = self.height * 1.0 / h

        np_labels = [[0, 0, 0, 0, 0]] * self.max_objects
        labels = [line.rstrip().split(' ') for line in open(label_file)]
        object_num = 0
        random.shuffle(labels)
        for label in labels:
            if not (label[0] == 'Car' or label[0] == 'Van'):
                continue

            x1 = float(label[4])
            y1 = float(label[5])
            x2 = float(label[6])
            y2 = float(label[7])
            assert x1 < x2
            assert y1 < y2
            class_num = 1

            xcenter = (x1 + x2) * 1.0 / 2 * width_rate
            ycenter = (y1 + y2) * 1.0 / 2 * height_rate

            box_w = (x2 - x1) * width_rate
            box_h = (y2 - y1) * height_rate

            np_labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
            object_num += 1
            if object_num >= self.max_objects:
                print("Object overflow")
                break

        return np_labels, object_num

    def record_process(self, record):
        """record process 
        Args: record 
        Returns:
          image: 3-D ndarray
          labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
          object_num:  total object number  int 
        """
        image_file, gt_image_file = record.split(" ")
        base_path = os.path.dirname(self.train_txt)
        image_file = os.path.join(base_path, image_file)
        gt_image_file = os.path.join(base_path, gt_image_file)

        image = scp.misc.imread(image_file)
        orig_shape = image.shape
        image = scp.misc.imresize(image, (self.height, self.width))

        labels, object_num = self.read_kitti_anno(gt_image_file,
                                                  orig_shape, False)

        return [image, labels, object_num]

    def record_customer(self):
        """record queue's customer 
        """
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        objects_num = []
        for i in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)
        images = np.asarray(images, dtype=np.float32)
        images = images/255 * 2 - 1
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num

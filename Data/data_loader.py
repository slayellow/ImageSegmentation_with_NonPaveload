import numpy as np
import os
import scipy.misc as misc
from PIL import Image
import random


class DataLoader:

    def __init__(self, image_dir, correct_label_dir, batch_size=1, shuffle=True):

        self.images = []
        self.labels = []
        self.num_files = 0
        self.epoch = 0
        self.itr = 0

        self.batch_size = batch_size
        self.image_dir = image_dir
        self.correct_label_dir = correct_label_dir

        self.ordered_files = []
        self.shuffle_files = []

        # 훈련 이미지 데이터 가져오기
        self.ordered_files += [each for each in os.listdir(self.image_dir) if
                              each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith(
                                  '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                                  '.tif') or each.endswith('.gif')]

        self.num_files = len(self.ordered_files)
        self.ordered_files.sort()

        # 파일을 랜덤하게 셔플
        if shuffle is True:
            self.shuffle_file()

    def shuffle_file(self):
        sf = np.array(range(np.int32(np.ceil(self.num_files / self.batch_size) + 1))) * self.batch_size
        random.shuffle(sf)

        for i in range(len(sf)):
            for k in range(self.batch_size):
                if sf[i] + k < self.num_files:
                    self.shuffle_files.append(self.ordered_files[sf[i] + k])

    def get_data(self):
        if self.itr >= self.num_files:
            self.itr = 0
            self.shuffle_file()
            self.epoch += 1
        batch_size = np.min([self.batch_size, self.num_files - self.itr])
        images = []
        for f in range(batch_size):
            with Image.open(self.image_dir + "/" + self.ordered_files[self.itr]) as Img:
                Img = np.array(Img)
                images.append(Img)
            self.itr += 1
        return np.array(images), self.ordered_files[self.itr-1]

    def get_data_label(self):
        self.images.clear()
        self.labels.clear()

        if self.itr >= self.num_files:
            self.itr = 0
            self.shuffle_file()
            self.epoch += 1
        batch_size = np.min([self.batch_size, self.num_files - self.itr])

        for f in range(batch_size):
            with Image.open(self.image_dir + "/" + self.shuffle_files[self.itr]) as Img:
                Img = np.array(Img)
                self.images.append(Img)

            with Image.open(self.correct_label_dir + "/" + self.shuffle_files[self.itr][0:-4] + ".png") as Label:
                Label = Label.convert('RGB')
                Label = np.array(Label)
                background_color = np.array([0, 0, 0])  # Black
                gt_bg = np.all(Label == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                self.labels.append(gt_image)

            self.itr += 1

            # 2021.01.28. 이미지 읽는 방식 수정으로 인한 주석
            # Img = misc.imread(self.image_dir + "/" + self.shuffle_files[self.itr])
            # Label = misc.imread(self.correct_label_dir + "/" + self.shuffle_files[self.itr][0:-4] + ".png")

            # Label = Label[:,:,0:3]
            # Img = Img[:, :, 0:3]

            # background_color = np.array([0, 0, 0])    # Black
            # sky_color = np.array([0, 128, 0])
            # road_color = np.array([128,0,0])

            # Create "one-hot-like" labels by class
            # gt_bg = np.all(Label == background_color, axis=2)
            # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)

            # 2021.01.27. 라벨 class 갯수 수정으로 인한 주석
            # gt_sky = np.all(Label == sky_color, axis=2)
            # gt_sky = gt_sky.reshape(*gt_sky.shape, 1)
            # gt_road = np.all(Label == road_color, axis=2)
            # gt_road = gt_road.reshape(*gt_road.shape, 1)

            # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            # 2021.01.27. 라벨 class 갯수 수정으로 인한 주석
            # gt_image = np.concatenate((gt_bg, gt_sky, gt_road), axis=2)

            # 2021.01.28. 이미지 읽는 방식 수정으로 인한 주석
            # images.append(Img)
            # labels.append(gt_image)

        return np.array(self.images), np.array(self.labels)



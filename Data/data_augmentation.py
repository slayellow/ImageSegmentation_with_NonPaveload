import cv2 as cv
import os


class DataAugmentation:

    def __init__(self, image_dir, correct_label_dir, save_dir):
        self.image_dir = image_dir
        self.correct_label_dir = correct_label_dir
        self.image_data_list = os.listdir(image_dir)
        self.correct_label_list = os.listdir(correct_label_dir)
        self.save_dir = save_dir
        print("Total Data Size : {}".format(len(self.image_data_list)))
        self.save_image_folder = "/augmentation_image"
        self.save_label_folder = "/augmentation_label"

    def save_image(self, save_dir, filename, img, mode, label=False):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if label is False:
            save_file = save_dir + "/%s%s.jpg" % (filename.split('.')[0], mode)
        else:
            save_file = save_dir + "/%s%s.png" % (filename.split('.')[0], mode)
        cv.imwrite(save_file, img)

    def make_augmentation(self):
        count = 1
        print("Data Augmentation Start ! ")
        for data in self.image_data_list:
            save_dir = self.save_dir + self.save_image_folder
            img = cv.imread(os.path.join(self.image_dir, data))
            # Flip
            self.flip(save_dir, data, img, 1, label=False)  # verical random flip
            # Rotation
            self.rotate(save_dir, data, img, 10, label=False)
            self.rotate(save_dir, data, img, 20, label=False)
            self.rotate(save_dir, data, img, -10, label=False)
            self.rotate(save_dir, data, img, -20, label=False)
            print(str(count) + " / " + str(len(self.image_data_list)) + " Data Augmentation Finish")
            count += 1
        print("Data Augmentation End ! ")
        print("---------------------------")
        print("Label Augmentation Start !")
        count = 1

        for label in self.correct_label_list:
            save_dir = self.save_dir + self.save_label_folder
            img = cv.imread(os.path.join(self.correct_label_dir, label))
            # Flip
            self.flip(save_dir, label, img, 1, label=True)
            # Rotation
            self.rotate(save_dir, label, img, 10, label=True)
            self.rotate(save_dir, label, img, 20, label=True)
            self.rotate(save_dir, label, img, -10, label=True)
            self.rotate(save_dir, label, img, -20, label=True)
            print(str(count) + " / " + str(len(self.correct_label_list)) + " Label Augmentation Finish")
            count += 1
        print("Label Augmentation End ! ")

    def flip(self, saved_dir, data, img, mode, label=False):
        img = cv.flip(img, mode)
        if mode == 0:
            mode = "_up_down_"
        elif mode == 1:
            mode = "_left_right_"
        elif mode == -1:
            mode = "_all_"
        self.save_image(saved_dir, data, img, mode, label)

    def rotate(self, saved_dir, data, img, rate=10, label=False):
        mode = "_rotate" + str(rate) + "_"
        x_length = img.shape[0]
        y_length = img.shape[1]
        rotation_matrix = cv.getRotationMatrix2D((x_length / 2, y_length / 2), rate, 1)
        img = cv.warpAffine(img, rotation_matrix, (y_length, x_length))
        self.save_image(saved_dir, data, img, mode, label)

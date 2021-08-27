import time
from multiprocessing import Process, Queue
from inference.DeepLab_V3 import DeepLab
from inference.DeepLab_V3_Plus import DeepLabV3Plus
from inference.pytorch_util import *
from inference.data_management import *
import inference.config as cf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix, PointCloud
from nav_msgs.msg import OccupancyGrid

import multiprocessing
import threading

import torch
from torchvision import transforms
import numpy as np
from timeit import default_timer as timer

import os
import cv2
from cv_bridge import CvBridge


class NonpaveloadSegmentor(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # GPU Check
        gpu_check = is_gpu_avaliable()
        devices = torch.device("cuda") if gpu_check else torch.device("cpu")

        self.model = DeepLab(101, cf.NUM_CLASSES).to(devices)
        # self.model = DeepLabV3Plus(cf.NUM_CLASSES).to(devices)
        self.model.eval()
        pretrained_path = cf.paths['pretrained_path']
        if os.path.isfile(os.path.join(pretrained_path, self.model.get_name() + '.pth')):
            print("Pretrained Model Open : ", self.model.get_name() + ".pth")
            checkpoint = load_weight_file(os.path.join(pretrained_path, self.model.get_name() + '.pth'))
            load_weight_parameter(self.model, checkpoint['state_dict'])
        else:
            print("No Pretrained Model")
            return

        self.map = OccupancyGrid()
        self.map.header.frame_id = cf.FRAME_ID
        self.map.info.width = cf.MAP_SIZE
        self.map.info.height = cf.MAP_SIZE
        self.map.info.resolution = 0.1
        self.map.info.origin.position.x = 15.0
        self.map.info.origin.position.y = 15.0
        self.map.info.origin.position.z = 0.0
        self.map.info.origin.orientation.x = 0.0
        self.map.info.origin.orientation.y = 0.0
        self.map.info.origin.orientation.z = 180.0

        self.image_subscriber = self.create_subscription(Image, cf.IMG_SUB, self.on_image, 10)
        print('Image Subscirber Create Success : ', cf.IMG_SUB)
        self._imagedata = None

        self.segmentation_publisher = self.create_publisher(Image, cf.SEGMENTATION_PUB, 100)
        print('Segmentation Image Publisher Create Success : ', cf.SEGMENTATION_PUB)

        self.probabilitymap_publisher = self.create_publisher(OccupancyGrid, cf.PROBMAP_PUB, 100)
        print("Probabilty Map Publisher Create Success : ", cf.PROBMAP_PUB)
        self.masked_cloud = None
        self.pts2d = None

        self.bridge = CvBridge()

    def get_probabilitymap(self, segmap):
        left_top = [185, 282]
        right_top = [309, 282]
        left_bottom = [98, 343]
        right_bottom = [299, 343]

        pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
        pts2 = np.float32([[120,234], [140, 234], [120, 254], [140, 254]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_result = cv2.warpPerspective(segmap, M, (cf.MAP_SIZE, cf.MAP_SIZE))

        m_probmap = np.array([100] * cf.MAP_SIZE * cf.MAP_SIZE, dtype=np.int8).reshape((cf.MAP_SIZE, cf.MAP_SIZE))
        m_probmap[np.where(img_result == 0)] = 0
        m_probmap[np.where(img_result == 1)] = 0
        m_probmap[np.where(img_result == 3)] = 0
        m_probmap[np.where(img_result == 4)] = 0

        return m_probmap

    def segmentation(self, img):
        torch.cuda.synchronize()
        transform = transforms.Compose([
            transforms.Resize((cf.IMG_WIDTH, cf.IMG_HEIGHT), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        tensor_to_image = transforms.ToPILImage()
        img = tensor_to_image(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t, 0)

        torch.cuda.synchronize()
        output = self.model(batch_t)
        torch.cuda.synchronize()
        pred = torch.argmax(output, axis=1)
        torch.cuda.synchronize()
        segmap = np.array(pred[0].cpu()).astype(np.uint8)
        torch.cuda.synchronize()
        segmap_class = segmap.copy()
        torch.cuda.synchronize()
        segmap = (decode_segmap(segmap) * 255).astype(np.uint8)
        torch.cuda.synchronize()
        return segmap, segmap_class

    def on_image(self, msg):
        img_data = np.asarray(msg.data)
        img = np.reshape(img_data, (msg.height, msg.width, 3))

        # Image Save
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imwrite("img.png", cv_image)

        start = timer()
        seg_map, seg_map_class = self.segmentation(img)

        pub_img = self.bridge.cv2_to_imgmsg(seg_map, "8UC3")
        pub_img.header.frame_id = cf.FRAME_ID
        self.segmentation_publisher.publish(pub_img)
        end = timer()
        print("Image -> Segmentation Image Send Time: ", (end - start))

        start = timer()

        prob_map = self.get_probabilitymap(seg_map_class)
        prob_map = np.fliplr(prob_map)
        # prob_map = cv2.GaussianBlur(prob_map, (5, 5), 0).astype(np.int8)

        self.map.data = np.reshape(prob_map, [-1]).tolist()
        self.probabilitymap_publisher.publish(self.map)

        end = timer()
        print('Segmentation Image -> Probabilty Map Send Time : ', (end - start))


def main(args=None):
    rclpy.init(args=args)
    inference = NonpaveloadSegmentor('nonpaveload_segmentor')
    rclpy.spin(inference)
    inference.multi_process.join()
    inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

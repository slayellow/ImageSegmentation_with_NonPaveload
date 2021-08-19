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


def get_RTMatrix():
    vector = cf.SE_VECTOR
    translation = vector[3:]
    rotation = np.array(vector[:3]).T
    rx = rotation[0] / 180 * np.pi
    ry = rotation[1] / 180 * np.pi
    rz = rotation[2] / 180 * np.pi

    matR = np.array([np.cos(rz) * np.cos(ry),
                     np.cos(rz) * np.sin(ry) * np.sin(rx) - np.sin(rz) * np.cos(rx),
                     np.cos(rz) * np.sin(ry) * np.cos(rx) + np.sin(rz) * np.sin(rx),
                     np.sin(rz) * np.cos(ry),
                     np.sin(rz) * np.sin(ry) * np.sin(rx) + np.cos(rz) * np.cos(rx),
                     np.sin(rz) * np.sin(ry) * np.cos(rx) - np.cos(rz) * np.sin(rx),
                     -(np.sin(ry)),
                     np.cos(ry) * np.sin(rx),
                     np.cos(ry) * np.cos(rx)])
    matR = np.reshape(matR, [3, 3])
    t = -np.matmul(matR.T, np.expand_dims(translation, 1))
    T = np.concatenate([matR.T, t], 1)
    prjMT = np.matmul(cf.INTRINSIC_PARAMETER, T)

    return prjMT


class NonpaveloadSegmentor(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Create a subscriber to the Image topic
        self.projectionRTMatrix = get_RTMatrix()

        # GPU Check
        gpu_check = is_gpu_avaliable()
        devices = torch.device("cuda") if gpu_check else torch.device("cpu")

        # self.model = DeepLab(101, cf.NUM_CLASSES).to(devices)
        self.model = DeepLabV3Plus(cf.NUM_CLASSES).to(devices)
        self.model.eval()
        pretrained_path = cf.paths['pretrained_path']
        if os.path.isfile(os.path.join(pretrained_path, self.model.get_name() + '.pth')):
            print("Pretrained Model Open : ", self.model.get_name() + ".pth")
            checkpoint = load_weight_file(os.path.join(pretrained_path, self.model.get_name() + '.pth'))
            load_weight_parameter(self.model, checkpoint['state_dict'])
        else:
            print("No Pretrained Model")
            return

        self.image_subscriber = self.create_subscription(Image, cf.IMG_SUB, self.on_image, 10)
        print('Image Subscirber Create Success : ', cf.IMG_SUB)
        self._imagedata = None
        self._segimg = None
        self._segimg_test = None
        self._segimg_mutex = threading.Lock()
        self._segimg_event = threading.Event()

        self.ins_subscriber = self.create_subscription(NavSatFix, cf.INS_SUB, self.on_ins, 100)
        print('INS Subscirber Create Success : ', cf.INS_SUB)
        self._insdata = NavSatFix()

        self.vlp_subscriber = self.create_subscription(PointCloud, cf.VELODYNE_SUB, self.on_vlp, 100)
        print("Velodyne Subscriber Create Success : ", cf.VELODYNE_SUB)
        self._vlp_count = 0
        self._vlpdata = None
        self._vlplist = []
        self._vlp_mutex = threading.Lock()
        self._vlp_event = threading.Event()

        self._xyz_mutex = threading.Lock()
        self._xyz_event = threading.Event()

        self.segmentation_publisher = self.create_publisher(Image, cf.SEGMENTATION_PUB, 100)
        print('Segmentation Image Publisher Create Success : ', cf.SEGMENTATION_PUB)

        self.probabilitymap_publisher = self.create_publisher(OccupancyGrid, cf.PROBMAP_PUB, 100)
        print("Probabilty Map Publisher Create Success : ", cf.PROBMAP_PUB)

        self.bridge = CvBridge()

        self.projection_thread = threading.Thread(target=self.project_xyz_to_img)
        self.projection_thread.start()

        self.pointcloud_thread = threading.Thread(target=self.convert_pointcloud_to_xyz)
        self.pointcloud_thread.start()

        # self.probmap_thread = threading.Thread(target=self.convert_segimg_to_probmap)
        # self.probmap_thread.start()

        # t = threading.Thread(target=self.on_thread)
        # t.start()

    def project_xyz_to_img(self):
        while self.projection_thread.is_alive():
            if self._xyz_event.wait(1):
                start = timer()
                self._xyz_mutex.acquire()
                xyz_list = self._vlpdata.copy()
                self._vlpdata = None
                self._xyz_mutex.release()

                ptr = np.dot(self.projectionRTMatrix, xyz_list.T)

                pts2d = (ptr / ptr[2, :])
                pts2d = pts2d[:-1, :].T

                selected_indexes = (pts2d[:, 0] > 0) & (pts2d[:, 0] < cf.IMG_HEIGHT) & (pts2d[:, 1] > 0) & (pts2d[:, 1] < cf.IMG_WIDTH)
                masked_cloud = xyz_list[selected_indexes]
                pts2d = pts2d[selected_indexes, :2]

                if self._imagedata is not None:
                    segimg = self._imagedata.copy()
                    red = [255, 255, 255]

                    for pts in pts2d:
                        segimg[int(pts[1]), int(pts[0])] = red

                    cv2.imshow('img', segimg)
                    cv2.waitKey(1)

                self._xyz_event.clear()

    def convert_pointcloud_to_xyz(self):
        while self.pointcloud_thread.is_alive():
            if self._vlp_event.wait(1):
                start = timer()
                self._vlp_mutex.acquire()
                pointcloud_list = self._vlplist.copy()
                self._vlplist = []
                self._vlp_mutex.release()

                pointcloud_list = [[pointcloud.x, pointcloud.y, pointcloud.z, 1] for pointcloud in pointcloud_list
                                   if pointcloud.y > 0]
                pointcloud_list = np.array(pointcloud_list)

                end = timer()
                print("PointCloud --> XYZ Array Time : ", (end-start))

                self._xyz_mutex.acquire()
                self._vlpdata = pointcloud_list
                self._xyz_mutex.release()

                self._xyz_event.set()

                self._vlp_event.clear()

    def convert_segimg_to_probmap(self):
        left_top = [211, 266]
        left_bottom = [83, 321]
        right_top = [259, 266]
        right_bottom = [272, 321]

        while self.probmap_thread.is_alive():
            if self._segimg_event.wait(1):
                self._segimg_event.clear()
                start = timer()
                self._segimg_mutex.acquire()
                segimg = self._segimg.copy()
                self._segimg_mutex.release()

                pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
                pts2 = np.float32([[0, 0], [cf.MAP_SIZE, 0], [0, cf.MAP_SIZE], [cf.MAP_SIZE, cf.MAP_SIZE]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_result = cv2.warpPerspective(segimg, M, (cf.MAP_SIZE, cf.MAP_SIZE))
                img_result = np.fliplr(img_result)

                probmap = self.prob_grid_map(img_result).astype(np.float)
                probmap = cv2.GaussianBlur(probmap, (5, 5), 0).astype(np.int8)

                ins = self.get_navigation()

                map = OccupancyGrid()
                map.header.frame_id = cf.FRAME_ID
                map.data = np.reshape(probmap, [-1]).tolist()
                map.info.width = cf.MAP_SIZE
                map.info.height = cf.MAP_SIZE
                map.info.resolution = 0.1
                map.info.origin.position.x = -15.0
                map.info.origin.position.y = -15.0
                map.info.origin.position.z = 0.0
                map.info.origin.orientation.x = 0.0
                map.info.origin.orientation.y = 0.0
                map.info.origin.orientation.z = 0.0

                self.probabilitymap_publisher.publish(map)
                end = timer()

                print('Segmentation Image -> Probabilty Map Send Time : ', (end - start))

    def on_thread(self):
        while True:
            img = self.get_image()
            if img is None:
                print("Image is None")
                continue
            else:
                start = timer()
                segmap, segmap_class = self.segmentation(img)

                pub_img = self.bridge.cv2_to_imgmsg(segmap, "8UC3")
                pub_img.header.frame_id = cf.FRAME_ID
                self.segmentation_publisher.publish(pub_img)
                end = timer()
                print("Image -> Segmentation Image Send Time: ", (end - start))

                self._segimg_mutex.acquire()
                self._segimg_test = segmap
                self._segimg = segmap_class
                self._segimg_mutex.release()

                self._segimg_event.set()

    def prob_grid_map(self, img):
        map_size = cf.MAP_SIZE

        m_pProbBlockImg = np.array([100] * map_size * map_size, dtype=np.int8).reshape((map_size, map_size))
        m_pProbBlockImg[np.where(img == 0)] = 0
        m_pProbBlockImg[np.where(img == 1)] = 0
        m_pProbBlockImg[np.where(img == 3)] = 0
        m_pProbBlockImg[np.where(img == 4)] = 0
        probability_map = m_pProbBlockImg
        return probability_map

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

    def on_vlp(self, msg):
        self._vlp_count += 1

        self._vlp_mutex.acquire()
        self._vlplist.extend(msg.points)
        self._vlp_mutex.release()

        if self._vlp_count > 10:
            self._vlp_event.set()
            self._vlp_count = 0

    def on_ins(self, msg):
        self._insdata = msg

    def on_image(self, msg):
        img_data = np.asarray(msg.data)
        img = np.reshape(img_data, (msg.height, msg.width, 3))
        self._imagedata = img

    def get_image(self):
        return self._imagedata

    def get_navigation(self):
        return self._insdata


def main(args=None):
    rclpy.init(args=args)
    inference = NonpaveloadSegmentor('nonpaveload_segmentor')
    rclpy.spin(inference)

    inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

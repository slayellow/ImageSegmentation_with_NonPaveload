import time
from multiprocessing import Process, Queue

import std_msgs.msg
from inference.DeepLab_V3 import DeepLab
from inference.DeepLab_V3_Plus import DeepLabV3Plus
from inference.pytorch_util import *
from inference.data_management import *
import inference.config as cf
import rclpy
from rclpy.node import Node
from std_msgs import *
from sensor_msgs.msg import Image, NavSatFix, PointCloud
from nav_msgs.msg import OccupancyGrid
from scipy.spatial import Delaunay
import multiprocessing
import threading

import torch
from torchvision import transforms
import numpy as np
from timeit import default_timer as timer
import networkx as nx
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
        self.projectionRTMatrix = get_RTMatrix()
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
        self.map.info.origin.position.y = 30.0
        self.map.info.origin.position.z = 0.0
        self.map.info.origin.orientation.x = 0.0
        self.map.info.origin.orientation.y = 0.0
        self.map.info.origin.orientation.z = 180.0

        self.image_subscriber = self.create_subscription(Image, cf.IMG_SUB, self.on_image, 100)
        print('Image Subscirber Create Success : ', cf.IMG_SUB)
        self._imagedata = None
        self.segmap = None

        self.vlp_subscriber = self.create_subscription(PointCloud, cf.VELODYNE_SUB, self.on_vlp, 100)
        print('Velodyne Subscriber Create Success : ', cf.VELODYNE_SUB)
        self._pts_mutex = threading.Lock()
        self._vlpdata = None
        self.pts2d = None
        self.masked_cloud = None

        self.segmentation_publisher = self.create_publisher(Image, cf.SEGMENTATION_PUB, 100)
        print('Segmentation Image Publisher Create Success : ', cf.SEGMENTATION_PUB)

        self.probabilitymap_publisher = self.create_publisher(OccupancyGrid, cf.PROBMAP_PUB, 100)
        print("Probabilty Map Publisher Create Success : ", cf.PROBMAP_PUB)
        self.masked_cloud = None
        self.pts2d = None

        self.bridge = CvBridge()

        self.on_timer()

        thread = threading.Thread(target=self.on_thread)
        thread.start()

    def on_thread(self):
        seg_map_class = None
        while True:
            start = timer()
            if self._imagedata is not None:
                img_data = np.asarray(self._imagedata.data)
                img = np.reshape(img_data, (self._imagedata.height, self._imagedata.width, 3))
                seg_map, seg_map_class = self.segmentation(img)
                pub_img = self.bridge.cv2_to_imgmsg(seg_map, "8UC3")
                pub_img.header.frame_id = cf.FRAME_ID
                self.segmentation_publisher.publish(pub_img)

            if self._vlpdata is not None:
                pointcloud_list = self._vlpdata.points
                pointcloud_list = [[pointcloud.x, pointcloud.y, pointcloud.z, 1] for pointcloud in pointcloud_list
                                   if pointcloud.y > 0]
                pointcloud_list = np.array(pointcloud_list)

                ptr = np.dot(self.projectionRTMatrix, pointcloud_list.T)
                pts2d = (ptr / ptr[2, :])
                pts2d = pts2d[:-1, :].T

                selected_indexes = (pts2d[:, 0] > 0) & (pts2d[:, 0] < cf.IMG_HEIGHT) & (pts2d[:, 1] > 0) & (
                        pts2d[:, 1] < cf.IMG_WIDTH)

                masked_cloud = pointcloud_list[selected_indexes]
                pts2d = pts2d[selected_indexes, :2]

                if seg_map_class is None:
                    continue
                else:
                    prob_map, road_points = self.get_probabilitymap(seg_map_class, pts2d, masked_cloud)

                    if len(road_points) > 50:
                        road_img_points_for_alphashapes = road_points[0: len(road_points): 10]

                        for k in range(len(road_points)):
                            if road_points[k][1] > 250:
                                road_img_points_for_alphashapes.append(road_points[k])

                        alfa = self.getAlfaShapes(road_img_points_for_alphashapes, [0.9])

                        for hull in alfa[0]:
                            cv2.fillPoly(prob_map, pts=[np.array(hull)], color=0)

                    new_probmap_without_gaussian = np.array([-1] * cf.MAP_SIZE * cf.MAP_SIZE, dtype=np.int8).reshape((
                        cf.MAP_SIZE, cf.MAP_SIZE))
                    new_probmap_without_gaussian[29:, ] = prob_map[:-29, :]
                    prob_map = new_probmap_without_gaussian

                    prob_map = np.fliplr(prob_map)
                    # prob_map = cv2.GaussianBlur(prob_map, (5, 5), 0).astype(np.int8)
                    self.map.data = np.reshape(prob_map, [-1]).tolist()
                    self.probabilitymap_publisher.publish(self.map)
                end = timer()
                print("Algorithm Processing Time : ", end-start)

    def sqrt_sum(self, a, b):
        x = (a[0] - b[0])
        y = (a[1] - b[1])
        return np.sqrt(x*x + y*y)

    def shapeToSomePolygons(self, shape):
        G = nx.Graph()
        allnodes = set()
        for line in shape:
            G.add_nodes_from(line)
            G.add_edge(line[0], line[1])
            allnodes.add(line[0])
            allnodes.add(line[1])

        result = []

        while allnodes:
            node = allnodes.pop()
            new_node = next(iter(G[node]), None)
            if not new_node:
                continue

            G.remove_edge(node, new_node)
            temp = nx.shortest_path(G, node, new_node)
            for j, t in enumerate(temp):
                if t in allnodes:
                    allnodes.remove(t)

            result.append(temp)

        return result

    def area_of_polygon_xy(self, x, y):
        area = 0.0
        for i in range(-1, len(x) - 1):
            area += x[i] * (y[i+1] - y[i-1])
        return abs(area) / 2.0

    def area_of_polygon_crd(self, cordinates):
        x = [v[0] for v in cordinates]
        y = [v[1] for v in cordinates]
        return self.area_of_polygon_xy(x, y)

    def getAlfaShapes(self, pts, alfas=[1]):
        tri_ind = [(0, 1), (1, 2), (2, 0)]
        tri = Delaunay(pts)
        lengths = {}
        for s in tri.simplices:
            for ind in tri_ind:
                a = pts[s[ind[0]]]
                b = pts[s[ind[1]]]
                line = (a, b)
                lengths[line] = self.sqrt_sum(a, b)
        ls = sorted(lengths.values())

        mean_length = np.mean(ls)
        mean_length_index = ls.index(next(filter(lambda x: x>=mean_length, ls)))
        magic_numbers = [ls[i] for i in range(mean_length_index, len(ls))]
        magic_numbers[0] = 0
        sum_magic = np.sum(magic_numbers)
        for i in range(2, len(magic_numbers)):
            magic_numbers[i] += magic_numbers[i-1]
        magic_numbers = [m / sum_magic for m in magic_numbers]

        rez = []

        for alfa in alfas:
            i = magic_numbers.index(next(filter(lambda z: z> alfa, magic_numbers), magic_numbers[-1]))
            av_length = ls[mean_length_index+i]

            lines = {}

            for s in tri.simplices:
                used = True
                for ind in tri_ind:
                    if lengths[(pts[s[ind[0]]], pts[s[ind[1]]])] > av_length:
                        used = False
                        break
                if used is False:
                    continue

                for ind in tri_ind:
                    i, j = s[ind[0]], s[ind[1]]
                    line = (pts[min(i, j)], pts[max(i, j)])
                    lines[line] = line in lines

            good_lines = []
            for v in lines:
                if not lines[v]:
                    good_lines.append(v)

            result = self.shapeToSomePolygons(good_lines)
            result.sort(key=self.area_of_polygon_crd, reverse=True)
            rez.append(result)
        return rez

    def on_timer(self):
        if self._imagedata is not None and self._vlpdata is not None:
            print("Image - VLP Timestamp --> ", self._imagedata.header.stamp,  self._vlpdata.header.stamp)
        else:
            print("No Receive Data")
        timer = threading.Timer(1, self.on_timer)
        timer.start()

    def get_probabilitymap(self, segmap, pts2d, masked_cloud):
        map_size = cf.MAP_SIZE
        map_center_x = int(map_size / 2)
        map_center_y = map_size
        grid_resolution = 10

        prob_map = np.array([-1] * map_size * map_size, dtype=np.int8).reshape((map_size, map_size))
        road_img_point = []

        for i in range(0, pts2d.shape[0], 1):
            pts2d_class = segmap[int(pts2d[i, 1]), int(pts2d[i, 0])]
            if pts2d_class == 0 or pts2d_class == 1 or pts2d_class == 3 or pts2d_class == 4:
                grid_x = map_center_x + int(masked_cloud[i, 0] * grid_resolution + 0.5)  # 0.5 ?
                grid_y = map_center_y - int(masked_cloud[i, 1] * grid_resolution + 0.5)  # 0.5 ?
                if 0 < grid_x < map_size and 0 < grid_y < map_size:
                    prob_map[grid_y, grid_x] = 0
                    road_img_point.append((grid_x, grid_y))
            else:
                grid_x = map_center_x + int(masked_cloud[i, 0] * grid_resolution + 0.5)  # 0.5 ?
                grid_y = map_center_y - int(masked_cloud[i, 1] * grid_resolution + 0.5)  # 0.5 ?
                if 0 < grid_x < map_size and 0 < grid_y < map_size:
                    prob_map[grid_y, grid_x] = 100

        return prob_map, road_img_point

    def on_vlp(self, msg):
        self._vlpdata = msg

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
        self._imagedata = msg


def main(args=None):
    rclpy.init(args=args)
    inference = NonpaveloadSegmentor('nonpaveload_segmentor')
    rclpy.spin(inference)
    inference.multi_process.join()
    inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

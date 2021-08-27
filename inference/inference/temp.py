def get_probabilitymap(self, segmap):
    # Split Road_Cloud / Rest_Cloud
    # Get Probability Map
    map_size = cf.MAP_SIZE
    map_center_x = int(map_size / 2)
    map_center_y = map_size
    grid_resolution = 10

    self._pts_mutex.acquire()
    pts2d = self.pts2d.copy()
    masked_cloud = self.masked_cloud.copy()

    self.pts2d = None
    self.masked_cloud = None
    self._pts_mutex.release()

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
                road_img_point.append((grid_x, grid_y))

    return prob_map, road_img_point


def on_vlp(self, msg):
    self.image_prev_time = time.time()
    self._vlp_count += 1
    self._vlplist.extend(msg.points)

    if self._vlp_count > 10:

        pointcloud_list = self._vlplist.copy()
        pointcloud_list = [[pointcloud.x, pointcloud.y, pointcloud.z, 1] for pointcloud in pointcloud_list
                           if pointcloud.y > 0]
        pointcloud_list = np.array(pointcloud_list)

        ptr = np.dot(self.projectionRTMatrix, pointcloud_list.T)
        pts2d = (ptr / ptr[2, :])
        pts2d = pts2d[:-1, :].T

        selected_indexes = (pts2d[:, 0] > 0) & (pts2d[:, 0] < cf.IMG_HEIGHT) & (pts2d[:, 1] > 0) & (
                    pts2d[:, 1] < cf.IMG_WIDTH)

        self._pts_mutex.acquire()
        self.masked_cloud = pointcloud_list[selected_indexes]
        self.pts2d = pts2d[selected_indexes, :2]
        self._pts_mutex.release()

        self._vlplist = []
        self._vlp_count = 0

        #### Projection Display
        # img = self.get_image()
        #
        # for i in range(0, self.pts2d.shape[0], 1):
        #     img[int(self.pts2d[i, 1]), int(self.pts2d[i, 0])] = 255
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(1)

def on_ins(self, msg):
    self._insdata = msg

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

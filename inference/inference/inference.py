import rclpy
from inference.DeepLab_V3 import DeepLab
from inference.pytorch_util import *
from inference.data_management import *
import inference.config as cf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Import necessary PyTorch and related frameworks
import torch
from torchvision import transforms
import numpy as np
from timeit import default_timer as timer

import os

import cv2
from cv_bridge import CvBridge, CvBridgeError


class NonpaveloadSegmentor(Node):
    def __init__(self):
        super().__init__('nonpaveload_segmentor')
        # Create a subscriber to the Image topic
        self.image_subscriber = self.create_subscription(Image, cf.IMG_SUB, self.listener_callback, 10)
        print('Image Subscirber Create Success : ', cf.IMG_SUB)

        # create a publisher onto the vision_msgs 2D classification topic
        self.segmentation_publisher = self.create_publisher(Image, cf.IMG_PUB, 10)
        print('Image Publisher Create Success : ', cf.IMG_PUB)

        # GPU Check
        gpu_check = is_gpu_avaliable()
        devices = torch.device("cuda") if gpu_check else torch.device("cpu")

        self.model = DeepLab(101, cf.NUM_CLASSES).to(devices)
        self.model.eval()
        pretrained_path = cf.paths['pretrained_path']
        if os.path.isfile(os.path.join(pretrained_path, self.model.get_name() + '.pth')):
            print("Pretrained Model Open : ", self.model.get_name() + ".pth")
            checkpoint = load_weight_file(os.path.join(pretrained_path, self.model.get_name() + '.pth'))
            load_weight_parameter(self.model, checkpoint['state_dict'])
        else:
            print("No Pretrained Model")
            return

        # Use CV bridge to convert ROS Image to CV_image for visualizing in window
        self.bridge = CvBridge()

    def segmentation(self, img):
        transform = transforms.Compose([
            transforms.Resize((cf.IMG_WIDTH, cf.IMG_HEIGHT), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        tensor_to_image = transforms.ToPILImage()
        img = tensor_to_image(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t, 0)

        # Classify the image
        start = timer()
        output = self.model(batch_t)
        end = timer()

        print("Segmentation Time: ", (end - start))

        pred = torch.argmax(output, axis=1)
        segmap = np.array(pred[0].cpu()).astype(np.uint8)
        segmap = (decode_segmap(segmap)*255).astype(np.uint8)
        return segmap

    def listener_callback(self, msg):

        img_data = np.asarray(msg.data)
        img = np.reshape(img_data, (msg.height, msg.width, 3))

        segmap = self.segmentation(img)

        pub_img = self.bridge.cv2_to_imgmsg(segmap,"8UC3")
        pub_img.header.frame_id = 'ImageCCD'

        self.segmentation_publisher.publish(pub_img)

        # Use OpenCV to visualize the images being classified from webcam
        # try:
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = segmap 
        # except CvBridgeError as e:
            # print(e)
        # cv2.imshow('result', cv_image)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    inference = NonpaveloadSegmentor()
    rclpy.spin(inference)

    inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

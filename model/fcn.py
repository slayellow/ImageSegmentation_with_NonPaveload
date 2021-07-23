import numpy as np
import tensorflow as tf

from . import npy_loader
from . import tensorflow_util


# Model FCN 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelFCN:


    def __init__(self, npy_path=None):
        self.vgg_mean = [103.939, 116.779, 123.68]

        self.npy = npy_loader.NPYLoader()
        self.npy.load_npy(npy_path)

    def build_model(self, image, classes, keep_prob):
        print("Build FCN Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.vgg_mean[0], green - self.vgg_mean[1], red - self.vgg_mean[2]])

        # Layer 1 ( conv3 -> conv3 -> maxpool )
        conv1_1 = self.make_conv_layer(bgr, "conv1_1")
        conv1_2 = self.make_conv_layer(conv1_1, "conv1_2")
        pool1 = tensorflow_util.set_max_pool(conv1_2)

        # Layer 2 ( conv3 -> conv3 -> maxpool )
        conv2_1 = self.make_conv_layer(pool1, "conv2_1")
        conv2_2 = self.make_conv_layer(conv2_1, "conv2_2")
        pool2 = tensorflow_util.set_max_pool(conv2_2)

        # Layer 3 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv3_1 = self.make_conv_layer(pool2, "conv3_1")
        conv3_2 = self.make_conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.make_conv_layer(conv3_2, "conv3_3")
        pool3 = tensorflow_util.set_max_pool(conv3_3)

        # Layer 4 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv4_1 = self.make_conv_layer(pool3, "conv4_1")
        conv4_2 = self.make_conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.make_conv_layer(conv4_2, "conv4_3")
        pool4 = tensorflow_util.set_max_pool(conv4_3)

        # Layer 5 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv5_1 = self.make_conv_layer(pool4, "conv5_1")
        conv5_2 = self.make_conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.make_conv_layer(conv5_2, "conv5_3")
        pool5 = tensorflow_util.set_max_pool(conv5_3)

        # FCL -> Fully Convolutional Layer 변경 ( 7 x 7 x 512 --> 1 x 1 x 4096 )
        # [7, 7, 512, 4096] --> [width, height, input channel, output channel]
        W6 = tensorflow_util.get_init_weight([7, 7, 512, 4096], name="W6")
        b6 = tensorflow_util.get_init_bias([4096], name="b6")
        conv6 = tensorflow_util.set_conv_layer(pool5, b6, W6)
        relu6 = tensorflow_util.set_relu(conv6, "relu6")
        dropout6 = tensorflow_util.set_dropout(relu6, keep_prob)

        # 1 x 1 Convolution
        W7 = tensorflow_util.get_init_weight([1, 1, 4096, 4096], name="W7")
        b7 = tensorflow_util.get_init_bias([4096], name="b7")
        conv7 = tensorflow_util.set_conv_layer(dropout6, b7, W7)
        relu7 = tensorflow_util.set_relu(conv7, name="relu7")
        dropout7 = tensorflow_util.set_dropout(relu7, keep_prob)

        # 1 x 1 Convolution
        W8 = tensorflow_util.get_init_weight([1, 1, 4096, classes], name="W8")
        b8 = tensorflow_util.get_init_bias([classes], name="b8")
        conv8 = tensorflow_util.set_conv_layer(dropout7, b8, W8)

        # Decoder - Upscale to Actual Image Size
        # 2x upsample
        upsampling_shape1 = pool4.get_shape()
        W_up1 = tensorflow_util.get_init_weight([4, 4, upsampling_shape1[3], classes], name="W_up1")
        b_up1 = tensorflow_util.get_init_bias([upsampling_shape1[3]], name="b_up1")
        conv_t1 = tensorflow_util.set_transpose_conv(conv8, b_up1, W_up1, output_shape=tf.shape(pool4))
        fuse_1 = tensorflow_util.set_add(conv_t1, pool4, name="fuse_1")

        # 2x upsample
        upsampling_shape2 = pool3.get_shape()
        W_up2 = tensorflow_util.get_init_weight([4, 4, upsampling_shape2[3], upsampling_shape1[3]], name="W_up2")
        b_up2 = tensorflow_util.get_init_bias([upsampling_shape2[3]], name="b_up2")
        conv_t2 = tensorflow_util.set_transpose_conv(fuse_1, b_up2, W_up2, output_shape=tf.shape(pool3))
        fuse_2 = tensorflow_util.set_add(conv_t2, pool3, name="fuse_2")

        # 8x upsample
        upsampling_shape3 = tf.shape(image)
        W_up3 = tensorflow_util.get_init_weight([16, 16, classes, upsampling_shape2[3]], name="W_up3")
        b_up3 = tensorflow_util.get_init_bias([classes], name="b_up3")
        fuse_3 = tensorflow_util.set_transpose_conv(fuse_2, b_up3, W_up3, output_shape=[upsampling_shape3[0],
                                                                                        upsampling_shape3[1],
                                                                                        upsampling_shape3[2],
                                                                                        classes], stride=8)

        print("FCN Model Build Success!")
        return fuse_3

    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):

        logits = tf.compat.v1.reshape(last_layer, (-1, num_classes), name="fcn_logits")
        correct_label_reshaped = tf.compat.v1.reshape(correct_label, (-1, num_classes))

        cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.compat.v1.reduce_mean(cross_entropy, name="fcn_loss")
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")
        return logits, train_op, loss_op

    def make_conv_layer(self, data, name):
        with tf.compat.v1.variable_scope(name):
            filt = tensorflow_util.get_conv_filter(self.npy.get_values(name, 0), name)
            bias = tensorflow_util.get_bias(self.npy.get_values(name, 1), name)

            conv = tensorflow_util.set_conv_layer(data, bias, filt)
            relu = tensorflow_util.set_relu(conv)

            return relu
import numpy as np
import tensorflow as tf

from . import npy_loader
from . import tensorflow_util


# Model DeepLab_v1 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelDeepLab_v1:


    def __init__(self, npy_path=None):
        self.vgg_mean = [103.939, 116.779, 123.68]

        self.npy = npy_loader.NPYLoader()
        self.npy.load_npy(npy_path)

    def build_model(self, image, classes, keep_prob):
        print("Build DeepLab_v1 Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.vgg_mean[0], green - self.vgg_mean[1], red - self.vgg_mean[2]])

        # Layer 1 ( conv3 -> conv3 -> maxpool )
        conv1_1 = self.make_conv_layer(bgr, "conv1_1")
        conv1_2 = self.make_conv_layer(conv1_1, "conv1_2")
        pool1 = tensorflow_util.set_max_pool(conv1_2, kernel=3)

        # Layer 2 ( conv3 -> conv3 -> maxpool )
        conv2_1 = self.make_conv_layer(pool1, "conv2_1")
        conv2_2 = self.make_conv_layer(conv2_1, "conv2_2")
        pool2 = tensorflow_util.set_max_pool(conv2_2, kernel=3)

        # Layer 3 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv3_1 = self.make_conv_layer(pool2, "conv3_1")
        conv3_2 = self.make_conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.make_conv_layer(conv3_2, "conv3_3")
        pool3 = tensorflow_util.set_max_pool(conv3_3, kernel=3)

        # Layer 4 ( conv3 -> conv3 -> conv3 -> maxpool(stride:1) )
        conv4_1 = self.make_conv_layer(pool3, "conv4_1")
        conv4_2 = self.make_conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.make_conv_layer(conv4_2, "conv4_3")
        pool4 = tensorflow_util.set_max_pool(conv4_3, kernel=3, stride=1)

        # Layer 5 ( conv3 -> conv3 -> conv3 -> maxpool(stride:1) -> avgpool(stride:1) )
        conv5_1 = self.make_atros_conv_layer(pool4, "conv5_1")
        conv5_2 = self.make_atros_conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.make_atros_conv_layer(conv5_2, "conv5_3")
        pool5 = tensorflow_util.set_max_pool(conv5_3, kernel=3, stride=1)
        avgpool = tensorflow_util.set_avg_pool(pool5, kernel=3, stride=1, name='avgpool')

        # 3 x 3 Convolution
        W6 = tensorflow_util.get_init_weight([3, 3, 512, 1024], name="W6")
        b6 = tensorflow_util.get_init_bias([1024], name="b6")
        conv6 = tensorflow_util.set_atros_conv_layer(avgpool, b6, W6, rate=12)
        relu6 = tensorflow_util.set_relu(conv6, "relu6")
        dropout6 = tensorflow_util.set_dropout(relu6, keep_prob)

        # 1 x 1 Convolution
        W7 = tensorflow_util.get_init_weight([1, 1, 1024, 1024], name="W7")
        b7 = tensorflow_util.get_init_bias([1024], name="b7")
        conv7 = tensorflow_util.set_conv_layer(dropout6, b7, W7)
        relu7 = tensorflow_util.set_relu(conv7, name="relu7")
        dropout7 = tensorflow_util.set_dropout(relu7, keep_prob)

        # 1 x 1 Convolution
        W8 = tensorflow_util.get_init_weight([1, 1, 1024, classes], name="W8")
        b8 = tensorflow_util.get_init_bias([classes], name="b8")
        conv8 = tensorflow_util.set_conv_layer(dropout7, b8, W8)

        upsampling_conv8 = tf.compat.v1.image.resize_bilinear(conv8, tf.shape(image)[1:3, ])

        print("DeepLab_v1 Model Build Success!")
        return upsampling_conv8

    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(last_layer, (-1, num_classes), name="deeplab_v1_logits")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.reduce_mean(cross_entropy, name="deeplab_v1_loss")

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="deeplab_v1_train_op")

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, 1)
        labels = tf.argmax(correct_label_reshaped, 1)

        mIoU, update_op = tensorflow_util.get_mIoU(pred_labels, labels, num_classes)

        return logits, train_op, loss_op, accuracy, mIoU, update_op

    def make_conv_layer(self, data, name):
        with tf.compat.v1.variable_scope(name):
            filt = tensorflow_util.get_conv_filter(self.npy.get_values(name, 0), name)
            bias = tensorflow_util.get_bias(self.npy.get_values(name, 1), name)
            conv = tensorflow_util.set_conv_layer(data, bias, filt)
            relu = tensorflow_util.set_relu(conv)

            return relu

    def make_atros_conv_layer(self, data, name):
        with tf.compat.v1.variable_scope(name):
            filt = tensorflow_util.get_conv_filter(self.npy.get_values(name, 0), name)
            bias = tensorflow_util.get_bias(self.npy.get_values(name, 1), name)
            conv = tensorflow_util.set_atros_conv_layer(data, bias, filt, rate=2)
            relu = tensorflow_util.set_relu(conv)

            return relu
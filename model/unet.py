import numpy as np
import tensorflow as tf

from . import npy_loader
from . import tensorflow_util


# Model UNet 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelUNet:

    def __init__(self):
        pass

    def build_model(self, image, classes, keep_prob, first=32):
        print("Build U-Net Model")

        # Encoder 1 --> filter first 64 -> 32 change
        W1 = tensorflow_util.get_init_weight([3, 3, 3, first], name="W1")
        b1 = tensorflow_util.get_init_bias([first], name="b1")
        conv1 = tensorflow_util.set_conv_layer(image, b1, W1)
        relu1 = tensorflow_util.set_relu(conv1, name="relu1")

        W2 = tensorflow_util.get_init_weight([3, 3, first, first], name="W2")
        b2 = tensorflow_util.get_init_bias([first], name="b2")
        conv2 = tensorflow_util.set_conv_layer(relu1, b2, W2)
        relu2 = tensorflow_util.set_relu(conv2, name="relu2")

        pool1 = tensorflow_util.set_max_pool(relu2)

        # Encoder 2
        W3 = tensorflow_util.get_init_weight([3, 3, first, first * 2], name="W3")
        b3 = tensorflow_util.get_init_bias([first * 2], name="b3")
        conv3 = tensorflow_util.set_conv_layer(pool1, b3, W3)
        relu3 = tensorflow_util.set_relu(conv3, name="relu3")

        W4 = tensorflow_util.get_init_weight([3, 3, first * 2, first * 2], name="W4")
        b4 = tensorflow_util.get_init_bias([first * 2], name="b4")
        conv4 = tensorflow_util.set_conv_layer(relu3, b4, W4)
        relu4 = tensorflow_util.set_relu(conv4, name="relu4")

        pool2 = tensorflow_util.set_max_pool(relu4)

        # Encoder 3
        W5 = tensorflow_util.get_init_weight([3, 3, first * 2, first * 4], name="W5")
        b5 = tensorflow_util.get_init_bias([first * 4], name="b5")
        conv5 = tensorflow_util.set_conv_layer(pool2, b5, W5)
        relu5 = tensorflow_util.set_relu(conv5, name="relu5")

        W6 = tensorflow_util.get_init_weight([3, 3, first * 4, first * 4], name="W6")
        b6 = tensorflow_util.get_init_bias([first * 4], name="b6")
        conv6 = tensorflow_util.set_conv_layer(relu5, b6, W6)
        relu6 = tensorflow_util.set_relu(conv6, name="relu6")

        pool3 = tensorflow_util.set_max_pool(relu6)

        # Encoder 4
        W7 = tensorflow_util.get_init_weight([3, 3, first * 4, first * 8], name="W7")
        b7 = tensorflow_util.get_init_bias([first * 8], name="b7")
        conv7 = tensorflow_util.set_conv_layer(pool3, b7, W7)
        relu7 = tensorflow_util.set_relu(conv7, name="relu7")

        W8 = tensorflow_util.get_init_weight([3, 3, first * 8, first * 8], name="W8")
        b8 = tensorflow_util.get_init_bias([first * 8], name="b8")
        conv8 = tensorflow_util.set_conv_layer(relu7, b8, W8)
        relu8 = tensorflow_util.set_relu(conv8, name="relu8")

        pool4 = tensorflow_util.set_max_pool(relu8)

        # Encoder 5 ( Final )
        W9 = tensorflow_util.get_init_weight([3, 3, first * 8, first * 16], name="W9")
        b9 = tensorflow_util.get_init_bias([first * 16], name="b9")
        conv9 = tensorflow_util.set_conv_layer(pool4, b9, W9)
        relu9 = tensorflow_util.set_relu(conv9, name="relu9")

        W10 = tensorflow_util.get_init_weight([3, 3, first * 16, first * 16], name="W10")
        b10 = tensorflow_util.get_init_bias([first * 16], name="b10")
        conv10 = tensorflow_util.set_conv_layer(relu9, b10, W10)
        relu10 = tensorflow_util.set_relu(conv10, name="relu10")

        # Decoder 1
        upsampling_shape1 = relu8.get_shape()
        W_up1 = tensorflow_util.get_init_weight([2, 2, upsampling_shape1[3], first * 16], name="W_up1")
        b_up1 = tensorflow_util.get_init_bias([upsampling_shape1[3]], name="b_up1")
        conv_t1 = tensorflow_util.set_transpose_conv(relu10, b_up1, W_up1, output_shape=tf.shape(relu8))
        merge1 = tensorflow_util.set_concat(relu8, conv_t1, name="merge1")

        W11 = tensorflow_util.get_init_weight([3, 3, first * 16, first * 8], name="W11")
        b11 = tensorflow_util.get_init_bias([first * 8], name="b11")
        conv11 = tensorflow_util.set_conv_layer(merge1, b11, W11)
        relu11 = tensorflow_util.set_relu(conv11, name="relu11")

        W12 = tensorflow_util.get_init_weight([3, 3, first * 8, first * 8], name="W12")
        b12 = tensorflow_util.get_init_bias([first * 8], name="b12")
        conv12 = tensorflow_util.set_conv_layer(relu11, b12, W12)
        relu12 = tensorflow_util.set_relu(conv12, name="relu12")
        dropout12 = tensorflow_util.set_dropout(relu12, keep_prob=keep_prob)

        # Decoder 2
        upsampling_shape2 = relu6.get_shape()
        W_up2 = tensorflow_util.get_init_weight([2, 2, upsampling_shape2[3], first * 8], name="W_up2")
        b_up2 = tensorflow_util.get_init_bias([upsampling_shape2[3]], name="b_up2")
        conv_t2 = tensorflow_util.set_transpose_conv(dropout12, b_up2, W_up2, output_shape=tf.shape(relu6))
        merge2 = tensorflow_util.set_concat(relu6, conv_t2, name="merge2")

        W13 = tensorflow_util.get_init_weight([3, 3, first * 8, first * 4], name="W13")
        b13 = tensorflow_util.get_init_bias([first * 4], name="b13")
        conv13 = tensorflow_util.set_conv_layer(merge2, b13, W13)
        relu13 = tensorflow_util.set_relu(conv13, name="relu13")

        W14 = tensorflow_util.get_init_weight([3, 3, first * 4, first * 4], name="W14")
        b14 = tensorflow_util.get_init_bias([first * 4], name="b14")
        conv14 = tensorflow_util.set_conv_layer(relu13, b14, W14)
        relu14 = tensorflow_util.set_relu(conv14, name="relu14")
        dropout14 = tensorflow_util.set_dropout(relu14, keep_prob=keep_prob)

        # Decoder 3
        upsampling_shape3 = relu4.get_shape()
        W_up3 = tensorflow_util.get_init_weight([2, 2, upsampling_shape3[3], first * 4], name="W_up3")
        b_up3 = tensorflow_util.get_init_bias([upsampling_shape3[3]], name="b_up3")
        conv_t3 = tensorflow_util.set_transpose_conv(dropout14, b_up3, W_up3, output_shape=tf.shape(relu4))
        merge3 = tensorflow_util.set_concat(relu4, conv_t3, name="merge3")

        W15 = tensorflow_util.get_init_weight([3, 3, first * 4, first * 2], name="W15")
        b15 = tensorflow_util.get_init_bias([first * 2], name="b15")
        conv15 = tensorflow_util.set_conv_layer(merge3, b15, W15)
        relu15 = tensorflow_util.set_relu(conv15, name="relu15")

        W16 = tensorflow_util.get_init_weight([3, 3, first * 2, first * 2], name="W16")
        b16 = tensorflow_util.get_init_bias([first * 2], name="b16")
        conv16 = tensorflow_util.set_conv_layer(relu15, b16, W16)
        relu16 = tensorflow_util.set_relu(conv16, name="relu16")
        dropout16 = tensorflow_util.set_dropout(relu16, keep_prob=keep_prob)

        # Decoder 4
        upsampling_shape4 = relu2.get_shape()
        W_up4 = tensorflow_util.get_init_weight([2, 2, upsampling_shape4[3], first * 2], name="W_up4")
        b_up4 = tensorflow_util.get_init_bias([upsampling_shape4[3]], name="b_up4")
        conv_t4 = tensorflow_util.set_transpose_conv(dropout16, b_up4, W_up4, output_shape=tf.shape(relu2))
        merge4 = tensorflow_util.set_concat(relu2, conv_t4, name="merge4")

        W17 = tensorflow_util.get_init_weight([3, 3, first * 2, first], name="W17")
        b17 = tensorflow_util.get_init_bias([first], name="b17")
        conv17 = tensorflow_util.set_conv_layer(merge4, b17, W17)
        relu17 = tensorflow_util.set_relu(conv17, name="relu17")

        W18 = tensorflow_util.get_init_weight([3, 3, first, first], name="W18")
        b18 = tensorflow_util.get_init_bias([first], name="b18")
        conv18 = tensorflow_util.set_conv_layer(relu17, b18, W18)
        relu18 = tensorflow_util.set_relu(conv18, name="relu18")

        # Final Layer
        W19 = tensorflow_util.get_init_weight([1, 1, first, classes], name="W19")
        b19 = tensorflow_util.get_init_bias([classes], name="b19")
        conv19 = tensorflow_util.set_conv_layer(relu18, b19, W19)

        print("U-Net Model Build Success!")

        return conv19


    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(last_layer, (-1, num_classes), name="unet_logits")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
        #                                             labels=tf.stop_gradient(correct_label_reshaped[:]), name="Loss")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.reduce_mean(cross_entropy, name="unet_loss")

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="unet_train_op")

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return logits, train_op, loss_op, accuracy

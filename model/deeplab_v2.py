import numpy as np
import tensorflow as tf
from . import npy_loader
from . import tensorflow_util


# Model DeepLab_v2 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelDeepLab_v2:

    def __init__(self):
        self.vgg_mean = [103.939, 116.779, 123.68]

    def build_model(self, image, classes, first=256):
        print("Build DeepLab_v2 Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.vgg_mean[0], green - self.vgg_mean[1], red - self.vgg_mean[2]])

        # conv1 -> kernel 7x7, channels 64, stride 2 --> Result : 1/2 scale
        conv1 = self.set_conv1_block(bgr)

        # conv2 -> maxpool 3x3, stride 2 --> Result : 1/4 scale
        #
        # [ kernel 1x1, channels 64     ]   stride : 1
        # [ kernel 3x3, channels 64     ]  x 3
        # [ kernel 1x1, channels 256    ]
        maxpool = tensorflow_util.set_max_pool(conv1, kernel=3)
        conv2 = self.set_residual_block(maxpool, first, 3, name="conv2")

        # conv3 --> Result : 1/8 scale
        #
        # [ kernel 1x1, channels 128    ]   stride : 2
        # [ kernel 3x3, channels 128    ]  x 4
        # [ kernel 1x1, channels 512    ]
        conv3 = self.set_residual_block(conv2, first*2, 4, is_half=True, name="conv3")

        # conv4 --> Result : 1/8 scale
        # Add Atrous Convolution
        # [ kernel 1x1, channels 256    ]   stride : 1
        # [ kernel 3x3, channels 256    ]  x 23
        # [ kernel 1x1, channels 1024    ]
        conv4 = self.set_atrous_residual_block(conv3, first*4, 23, 2, name="conv4")

        # conv5 --> Result : 1/8 scale
        # Add Atrous Convolution
        # [ kernel 1x1, channels 512    ]   stride : 1
        # [ kernel 3x3, channels 512    ]  x 3
        # [ kernel 1x1, channels 2048    ]
        conv5 = self.set_atrous_residual_block(conv4, first*8, 3, 4, name="conv5")

        # ASPP [6, 12, 18, 24]
        #
        # Atrous Conv   3x3
        # Conv          1x1
        # Conv          1x1
        output = self.set_ASPP(conv5, classes=classes, atrous_list=[6, 12, 18, 24])

        upsampling_output = tf.compat.v1.image.resize_bilinear(output, tf.shape(image)[1:3, ])

        print("DeepLab_v2 Model Build Success!")
        return upsampling_output

    def set_conv1_block(self, data):
        W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], 64], name="conv1_W1")
        b1 = tensorflow_util.get_init_bias([64], name="conv1_b1")
        conv1 = tensorflow_util.set_conv_layer(data, b1, W1, strides=2)
        bn1 = tensorflow_util.set_batch_normalization(conv1, name='conv1_bn1')
        relu1 = tensorflow_util.set_relu(bn1, name="relu1")
        return relu1

    def set_residual_block(self, data, channel, block_num, is_half=False, name="residual"):
        first = 2 if is_half else 1
        data_channel = data.shape[3]
        layer_data = data
        for i in range(block_num):
            if i==0:
                W0 = tensorflow_util.get_init_weight([1, 1, data_channel, channel], name=name+"_residual_"+str(i)+"W0")
                b0 = tensorflow_util.get_init_weight([channel], name=name+"_residual_"+str(i)+"b0")
                conv0 = tensorflow_util.set_conv_layer(layer_data, b0, W0, strides=first)
                skip = tensorflow_util.set_batch_normalization(conv0, name=name+"_residual_"+str(i)+"bn0")
            else:
                skip = layer_data

            W1 = tensorflow_util.get_init_weight([1, 1, data_channel, (int)(channel / 4)],
                                                 name=name+"_residual_"+str(i)+"W1")
            b1 = tensorflow_util.get_init_bias([channel / 4], name=name+"_residual_"+str(i)+"b1")
            conv1 = tensorflow_util.set_conv_layer(layer_data, b1, W1, strides=first)
            bn1 = tensorflow_util.set_batch_normalization(conv1, name=name+"_residual_"+str(i)+"bn1")
            relu1 = tensorflow_util.set_relu(bn1, name=name+"_residual_"+str(i)+"relu1")

            W2 = tensorflow_util.get_init_weight([3, 3, relu1.shape[3], (int)(channel / 4)],
                                                 name=name + "_residual_" + str(i) + "W2")
            b2 = tensorflow_util.get_init_bias([channel / 4], name=name + "_residual_" + str(i) + "b2")
            conv2 = tensorflow_util.set_conv_layer(relu1, b2, W2)
            bn2 = tensorflow_util.set_batch_normalization(conv2, name=name + "_residual_" + str(i) + "bn2")
            relu2 = tensorflow_util.set_relu(bn2, name=name + "_residual_" + str(i) + "relu2")

            W3 = tensorflow_util.get_init_weight([1, 1, relu2.shape[3], channel],
                                                 name=name + "_residual_" + str(i) + "W3")
            b3 = tensorflow_util.get_init_bias([channel], name=name + "_residual_" + str(i) + "b3")
            conv3 = tensorflow_util.set_conv_layer(relu2, b3, W3)
            bn3 = tensorflow_util.set_batch_normalization(conv3, name=name + "_residual_" + str(i) + "bn3")

            add = tensorflow_util.set_add(skip, bn3, name=name+"_residual_"+str(i)+"add")
            relu3 = tensorflow_util.set_relu(add, name=name + "_residual_" + str(i) + "relu3")

            data_channel = relu3.shape[3]
            layer_data = relu3
            first = 1

        return layer_data

    def set_atrous_residual_block(self, data, channel, block_num, atrous=2, name="residual"):
        data_channel = data.shape[3]
        layer_data = data
        for i in range(block_num):
            if i==0:
                W0 = tensorflow_util.get_init_weight([1, 1, data_channel, channel], name=name+"_residual_"+str(i)+"W0")
                b0 = tensorflow_util.get_init_weight([channel], name=name+"_residual_"+str(i)+"b0")
                conv0 = tensorflow_util.set_conv_layer(layer_data, b0, W0)
                skip = tensorflow_util.set_batch_normalization(conv0, name=name+"_residual_"+str(i)+"bn0")
            else:
                skip = layer_data

            W1 = tensorflow_util.get_init_weight([1, 1, data_channel, (int)(channel / 4)],
                                                 name=name+"_residual_"+str(i)+"W1")
            b1 = tensorflow_util.get_init_bias([channel / 4], name=name+"_residual_"+str(i)+"b1")
            conv1 = tensorflow_util.set_conv_layer(layer_data, b1, W1)
            bn1 = tensorflow_util.set_batch_normalization(conv1, name=name+"_residual_"+str(i)+"bn1")
            relu1 = tensorflow_util.set_relu(bn1, name=name+"_residual_"+str(i)+"relu1")

            W2 = tensorflow_util.get_init_weight([3, 3, relu1.shape[3], (int)(channel / 4)],
                                                 name=name + "_residual_" + str(i) + "W2")
            b2 = tensorflow_util.get_init_bias([channel / 4], name=name + "_residual_" + str(i) + "b2")
            conv2 = tensorflow_util.set_atros_conv_layer(relu1, b2, W2, rate=atrous)
            bn2 = tensorflow_util.set_batch_normalization(conv2, name=name + "_residual_" + str(i) + "bn2")
            relu2 = tensorflow_util.set_relu(bn2, name=name + "_residual_" + str(i) + "relu2")

            W3 = tensorflow_util.get_init_weight([1, 1, relu2.shape[3], channel],
                                                 name=name + "_residual_" + str(i) + "W3")
            b3 = tensorflow_util.get_init_bias([channel], name=name + "_residual_" + str(i) + "b3")
            conv3 = tensorflow_util.set_conv_layer(relu2, b3, W3)
            bn3 = tensorflow_util.set_batch_normalization(conv3, name=name + "_residual_" + str(i) + "bn3")

            add = tensorflow_util.set_add(skip, bn3, name=name+"_residual_"+str(i)+"add")
            relu3 = tensorflow_util.set_relu(add, name=name + "_residual_" + str(i) + "relu3")

            data_channel = relu3.shape[3]
            layer_data = relu3

        return layer_data

    def set_ASPP(self, data, classes, atrous_list=[6, 12, 18, 24]):
        output = []
        for i, atrous in enumerate(atrous_list):
            W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], (int)(data.shape[3]*2)],
                                                 name="ASPP"+ str(i) + "W1")
            b1 = tensorflow_util.get_init_bias([(int)(data.shape[3]*2)], name="ASPP"+ str(i) + "b1")
            conv1 = tensorflow_util.set_atros_conv_layer(data, b1, W1, rate=atrous)
            bn1 = tensorflow_util.set_batch_normalization(conv1, name="ASPP"+ str(i) + "bn1")
            relu1 = tensorflow_util.set_relu(bn1, name="ASPP"+ str(i) +  "relu1")

            W2 = tensorflow_util.get_init_weight([1, 1, relu1.shape[3], relu1.shape[3]],
                                                 name="ASPP"+ str(i) + "W2")
            b2 = tensorflow_util.get_init_bias([relu1.shape[3]], name="ASPP"+ str(i) + "b2")
            conv2 = tensorflow_util.set_conv_layer(relu1, b2, W2)
            bn2 = tensorflow_util.set_batch_normalization(conv2, name="ASPP"+ str(i) + "bn2")
            relu2 = tensorflow_util.set_relu(bn2, name="ASPP"+ str(i) +  "relu2")

            W3 = tensorflow_util.get_init_weight([1, 1, relu2.shape[3], classes],
                                                 name="ASPP"+ str(i) + "W3")
            b3 = tensorflow_util.get_init_bias([classes], name="ASPP"+ str(i) + "b3")
            conv3 = tensorflow_util.set_conv_layer(relu2, b3, W3)
            output.append(conv3)
        return tensorflow_util.set_add_list(output, name="deeplab_v2_output")

    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(last_layer, (-1, num_classes), name="deeplab_v2_logits")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.reduce_mean(cross_entropy, name="deeplab_v2_loss")

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op,
                                                                                          name="deeplab_v2_train_op")

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, 1)
        labels = tf.argmax(correct_label_reshaped, 1)

        mIoU, update_op = tensorflow_util.get_mIoU(pred_labels, labels, num_classes)

        return logits, train_op, loss_op, accuracy, mIoU, update_op

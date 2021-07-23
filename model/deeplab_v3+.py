import numpy as np
import tensorflow as tf
from . import tensorflow_util


# Model DeepLab_v3+ 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelDeepLab_v3_Plus:

    def __init__(self):
        self.mean = [103.939, 116.779, 123.68]

    def build_model(self, image, classes, first=32):
        print("Build DeepLab_v3+ Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.mean[0], green - self.mean[1], red - self.mean[2]])

        # Entry Flow
        # Conv 32, 3x3, stride 2    : 1/2
        # Conv 64, 3x3
        # Separable Block ( 128, 3x3 --> 128, 3x3 --> 128, 3x3 stride 2) + Conv 128, 1x1, stride 2  : 1/4
        # Separable Block ( 256, 3x3 --> 256, 3x3 --> 256, 3x3 stride 2) + Conv 256, 1x1, stride 2  : 1/8
        # Separable Block ( 512, 3x3 --> 512, 3x3 --> 512, 3x3 stride 2) + Conv 512, 1x1, stride 2  : 1/16

        # Middle Flow ( x 16 )
        # Separable Conv 512, 3x3
        # Separable Conv 512, 3x3
        # Separable Conv 512, 3x3
        # Skip Connection ( Add Before Layer )

        # Exit Flow
        # Separable Block ( 512, 3x3 --> 1024, 3x3 --> 1024, 3x3) + Conv 1024, 1x1
        # Separable Conv 2048, 3x3
        # Separable Conv 2048, 3x3
        # Separable Conv 2048, 3x3

        # ASPP --> [6, 12, 18]
        #
        # Concat Global Average Pooling + ASPP + 1x1 Convolution
        # 1x1 Convolution
        output = self.set_ASPP(conv5, classes=classes, atrous_list=[6,12,18], depth=first)

        # Decoder
        # output --> X4 Upsample ( 256 )
        # Separable Block ( 128 ) --> Conv 48, 1x1
        # Concat 256 + 48
        # Conv 256, 3x3
        # Conv 256, 3x3
        # Conv classes, 1x1

        # upsampling_output = tf.compat.v1.image.resize_bilinear(output, tf.shape(image)[1:3, ])
        upsampling_output = tensorflow_util.set_bilinear_upsampling(output, tf.shape(image)[1:3, ])

        print("DeepLab_v3+ Model Build Success!")
        return upsampling_output

    def set_ASPP(self, data, classes, atrous_list=[6, 12, 18], depth=256):
        output = []

        image_level_features = tensorflow_util.set_global_average_pooling(data, name='global_average_pool')
        W0 = tensorflow_util.get_init_weight([1,1 , data.shape[3], depth],name="image_level_W0")
        b0 = tensorflow_util.get_init_bias([depth], name="image_level_b0")
        image_level_features = tensorflow_util.set_conv_layer(image_level_features, b0, W0)
        image_level_features = tensorflow_util.set_batch_normalization(image_level_features, name="image_level_bn0")
        image_level_features = tensorflow_util.set_bilinear_upsampling(image_level_features, (tf.shape(data)[1], tf.shape(data)[2]))
        output.append(image_level_features)

        aspp_W = tensorflow_util.get_init_weight([1,1, data.shape[3], depth], name="ASPP_W")
        aspp_b = tensorflow_util.get_init_bias([depth], name="ASPP_b")
        aspp_conv_1x1 = tensorflow_util.set_conv_layer(data, aspp_b, aspp_W)
        aspp_conv_1x1 = tensorflow_util.set_batch_normalization(aspp_conv_1x1, "ASPP_bn")
        output.append(aspp_conv_1x1)
        for i, atrous in enumerate(atrous_list):
            W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], depth], name="ASPP"+ str(i) + "W1")
            b1 = tensorflow_util.get_init_bias([depth], name="ASPP"+ str(i) + "b1")
            conv1 = tensorflow_util.set_atros_conv_layer(data, b1, W1, rate=atrous)
            conv1 = tensorflow_util.set_batch_normalization(conv1, name="ASPP"+ str(i) + "bn1")
            output.append(conv1)

        aspp_concat = tensorflow_util.set_concat(output, axis=3, name="ASPP_Concat")

        aspp_output_W = tensorflow_util.get_init_weight([1, 1, aspp_concat.shape[3], depth], name="ASPP_output_W")
        aspp_output_b = tensorflow_util.get_init_bias([depth], name="ASPP_output_b")
        aspp_output_conv = tensorflow_util.set_conv_layer(aspp_concat, aspp_output_b, aspp_output_W)
        aspp_output_conv = tensorflow_util.set_batch_normalization(aspp_output_conv, "ASPP_output_bn")

        output_W = tensorflow_util.get_init_weight([1, 1, aspp_output_conv.shape[3], classes], name="output_W")
        output_b = tensorflow_util.get_init_bias([classes], name="output_b")
        output = tensorflow_util.set_conv_layer(aspp_output_conv, output_b, output_W)

        return output

    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(last_layer, (-1, num_classes), name="deeplab_v3_plus_logits")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.reduce_mean(cross_entropy, name="deeplab_v3_plus_loss")

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op,
                                                                                          name="deeplab_v3_plus_train_op")

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, 1)
        labels = tf.argmax(correct_label_reshaped, 1)

        mIoU, update_op = tensorflow_util.get_mIoU(pred_labels, labels, num_classes)

        return logits, train_op, loss_op, accuracy, mIoU, update_op

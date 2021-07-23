import numpy as np
import tensorflow as tf
from . import tensorflow_util


# Model DeepLab_v3+ 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelDeepLab_v3_Plus:

    def __init__(self):
        self.mean = [103.939, 116.779, 123.68]

    def build_model(self, image, classes, first=32, flow=16):
        print("Build DeepLab_v3+ Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.mean[0], green - self.mean[1], red - self.mean[2]])

        # Entry Flow
        # Conv 32, 3x3, stride 2    : 1/2
        # Conv 64, 3x3
        # Separable Block ( 128, 3x3 --> 128, 3x3 --> 128, 3x3 stride 2) + Conv 128, 1x1, stride 2  : 1/4
        # Separable Block ( 256, 3x3 --> 256, 3x3 --> 256, 3x3 stride 2) + Conv 256, 1x1, stride 2  : 1/8
        # Separable Block ( 512, 3x3 --> 512, 3x3 --> 512, 3x3 stride 2) + Conv 512, 1x1, stride 2  : 1/16
        entry = self.set_entry_flow(bgr, channels=first)

        # Middle Flow ( x 16 )
        # Separable Conv 512, 3x3
        # Separable Conv 512, 3x3
        # Separable Conv 512, 3x3
        # Skip Connection ( Add Before Layer )
        middle = self.set_middle_flow(entry, channels=first * 16, flow=flow)

        # Exit Flow
        # Separable Block ( 512, 3x3 --> 1024, 3x3 --> 1024, 3x3) + Conv 1024, 1x1
        # Separable Conv 2048, 3x3
        # Separable Conv 2048, 3x3
        # Separable Conv 2048, 3x3
        exit = self.set_exit_flow(middle, channels=first * 32, strides=1, rate=1)

        # ASPP --> [6, 12, 18]
        #
        # Concat Global Average Pooling + ASPP + 1x1 Convolution
        # 1x1 Convolution
        aspp = self.set_ASPP(exit, classes=classes, atrous_list=[6,12,18], depth=first*8)

        # Decoder
        # output --> X4 Upsample ( 256 )
        # Separable Block ( 128 ) --> Conv 48, 1x1
        # Concat 128 + 48
        # Conv 128, 3x3
        # Conv 128, 3x3
        # Conv classes, 1x1
        output = self.set_decoder(aspp, num_class=classes)

        # upsampling_output = tf.compat.v1.image.resize_bilinear(output, tf.shape(image)[1:3, ])
        upsampling_output = tensorflow_util.set_bilinear_upsampling(output, tf.shape(image)[1:3, ])

        print("DeepLab_v3+ Model Build Success!")
        return upsampling_output

    def set_decoder(self, data, num_class ):
        upsample_list = []
        upsample_data = tensorflow_util.set_bilinear_upsampling(data, tf.shape(self.block1)[1:3, ])
        upsample_list.append(upsample_data)

        W0 = tensorflow_util.get_init_weight([1, 1, self.block1.shape[3], 48], name='decoder_W0')
        b0 = tensorflow_util.get_init_weight([48], name='decoder_b0')
        conv0 = tensorflow_util.set_conv_layer(self.block1, b0, W0)
        bn0 = tensorflow_util.set_batch_normalization(conv0, name='decoder_bn0')
        relu0 = tensorflow_util.set_relu(bn0, name='decoder_relu0')
        upsample_list.append(relu0)

        concat = tensorflow_util.set_concat(upsample_list, axis=3, name='docoder_concat')

        W1 = tensorflow_util.get_init_weight([3, 3, concat.shape[3], self.block1.shape[3]], name='decoder_W1')
        b1 = tensorflow_util.get_init_weight([self.block1.shape[3]], name='decoder_b1')
        conv1 = tensorflow_util.set_conv_layer(concat, b1, W1)
        bn1 = tensorflow_util.set_batch_normalization(conv1, name='decoder_bn1')
        relu1 = tensorflow_util.set_relu(bn1, name='decoder_relu1')

        W2 = tensorflow_util.get_init_weight([1, 1, relu1.shape[3], relu1.shape[3]], name='decoder_W2')
        b2 = tensorflow_util.get_init_weight([relu1.shape[3]], name='decoder_b2')
        conv2 = tensorflow_util.set_conv_layer(relu1, b2, W2)
        bn2 = tensorflow_util.set_batch_normalization(conv2, name='decoder_bn2')
        relu2 = tensorflow_util.set_relu(bn2, name='decoder_relu2')

        output_W = tensorflow_util.get_init_weight([1, 1, relu2.shape[3], num_class], name="output_W")
        output_b = tensorflow_util.get_init_bias([num_class], name="output_b")
        output = tensorflow_util.set_conv_layer(relu2, output_b, output_W)

        return output

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
        aspp_output_conv = tensorflow_util.set_relu(aspp_output_conv, name="ASPP_output_relu")

        return aspp_output_conv

    def set_entry_flow(self, data, channels):
        W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], channels], name='conv1_W1')
        b1 = tensorflow_util.get_init_bias([channels], name='conv1_b1')
        conv1 = tensorflow_util.set_conv_layer(data, b1, W1, strides=2)
        bn1 = tensorflow_util.set_batch_normalization(conv1, name='conv1_bn1')
        relu1 = tensorflow_util.set_relu(bn1, name='conv1_relu1')

        W2 = tensorflow_util.get_init_weight([3, 3, channels, channels*2], name='conv2_W2')
        b2 = tensorflow_util.get_init_bias([channels*2], name='conv2_b2')
        conv2 = tensorflow_util.set_conv_layer(relu1, b2, W2, strides=1)
        bn2 = tensorflow_util.set_batch_normalization(conv2, name='conv2_bn2')
        relu2 = tensorflow_util.set_relu(bn2, name='conv2_relu2')

        self.block1 = self.set_separable_block(relu2, channels=channels*4, residual_conv=True, strides=2, rate=1, name='block1')
        block2 = self.set_separable_block(self.block1, channels=channels*8, residual_conv=True, strides=2, rate=1, name='block2')
        block3 = self.set_separable_block(block2, channels=channels*16, residual_conv=True, strides=2, rate=1, name='block3')
        return block3

    def set_middle_flow(self, data, channels, flow):
        before_data = data
        for i in range(flow):
            block = self.set_separable_block(before_data, channels=channels, residual_conv=False, name='block'+str(i+4))
            before_data = block

        return before_data

    def set_exit_flow(self, data, channels, strides=1, rate=1):
        residual_W = tensorflow_util.get_init_weight([1, 1, data.shape[3], channels], name='exit_residual_W')
        residual_b = tensorflow_util.get_init_bias([channels], name='exit_residual_b')
        residual_conv = tensorflow_util.set_conv_layer(data, residual_b, residual_W, strides=strides)
        residual_bn = tensorflow_util.set_batch_normalization(residual_conv, name='exit_residual_bn')

        depth_W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], 1], name='exit_depth_W1')
        point_W1 = tensorflow_util.get_init_weight([1, 1, 1 * data.shape[3], (int)(channels/2)], name='exit_point_W1')
        b1 = tensorflow_util.get_init_bias([int(channels/2)], name="exit_b1")
        conv1 = tensorflow_util.set_depthwise_separable_conv_layer(data, b1, depth_W1, point_W1, strides=1, rate=rate)
        bn1 = tensorflow_util.set_batch_normalization(conv1, name='exit_bn1')
        relu1 = tensorflow_util.set_relu(bn1, name='exit_relu1')

        depth_W2 = tensorflow_util.get_init_weight([3, 3, relu1.shape[3], 1], name='exit_depth_W2')
        point_W2 = tensorflow_util.get_init_weight([1, 1, 1 * relu1.shape[3], channels], name='exit_point_W2')
        b2 = tensorflow_util.get_init_bias([channels], name="exit_b2")
        conv2 = tensorflow_util.set_depthwise_separable_conv_layer(relu1, b2, depth_W2, point_W2, strides=1, rate=rate)
        bn2 = tensorflow_util.set_batch_normalization(conv2, name='exit_bn2')
        relu2 = tensorflow_util.set_relu(bn2, name='exit_relu2')

        depth_W3 = tensorflow_util.get_init_weight([3, 3, relu2.shape[3], 1], name='exit_depth_W3')
        point_W3 = tensorflow_util.get_init_weight([1, 1, 1 * relu2.shape[3], channels], name='exit_point_W3')
        b3 = tensorflow_util.get_init_bias([channels], name="exit_b3")
        conv3 = tensorflow_util.set_depthwise_separable_conv_layer(relu2, b3, depth_W3, point_W3, strides=strides, rate=rate)
        bn3 = tensorflow_util.set_batch_normalization(conv3, name='exit_bn3')

        skip = tensorflow_util.set_add(residual_bn, bn3, name='exit_add')
        relu3 = tensorflow_util.set_relu(skip, name='exit_relu3')

        depth_W4 = tensorflow_util.get_init_weight([3, 3, relu3.shape[3], 1], name='exit_depth_W4')
        point_W4 = tensorflow_util.get_init_weight([1, 1, 1 * relu3.shape[3], channels*2], name='exit_point_W4')
        b4 = tensorflow_util.get_init_bias([channels*2], name="exit_b4")
        conv4 = tensorflow_util.set_depthwise_separable_conv_layer(relu3, b4, depth_W4, point_W4, strides=1, rate=rate*2)
        bn4 = tensorflow_util.set_batch_normalization(conv4, name='exit_bn4')
        relu4 = tensorflow_util.set_relu(bn4, name='exit_relu4')

        depth_W5 = tensorflow_util.get_init_weight([3, 3, relu4.shape[3], 1], name='exit_depth_W5')
        point_W5 = tensorflow_util.get_init_weight([1, 1, 1 * relu4.shape[3], channels*2], name='exit_point_W5')
        b5 = tensorflow_util.get_init_bias([channels*2], name="exit_b5")
        conv5 = tensorflow_util.set_depthwise_separable_conv_layer(relu4, b5, depth_W5, point_W5, strides=1, rate=rate*2)
        bn5 = tensorflow_util.set_batch_normalization(conv5, name='exit_bn5')
        relu5 = tensorflow_util.set_relu(bn5, name='exit_relu5')

        depth_W6 = tensorflow_util.get_init_weight([3, 3, relu5.shape[3], 1], name='exit_depth_W6')
        point_W6 = tensorflow_util.get_init_weight([1, 1, 1 * relu5.shape[3], channels*2], name='exit_point_W6')
        b6 = tensorflow_util.get_init_bias([channels*2], name="exit_b6")
        conv6 = tensorflow_util.set_depthwise_separable_conv_layer(relu5, b6, depth_W6, point_W6, strides=1, rate=rate*2)
        bn6 = tensorflow_util.set_batch_normalization(conv6, name='exit_bn6')
        relu6 = tensorflow_util.set_relu(bn6, name='exit_relu6')
        return relu6

    def set_separable_block(self, data, channels, residual_conv=True, strides=1, rate=1, name='block'):
        if residual_conv is True:
            residual_W = tensorflow_util.get_init_weight([1, 1, data.shape[3], channels], name=name+'_residual_W')
            residual_b = tensorflow_util.get_init_bias([channels], name=name+'_residual_b')
            residual_conv = tensorflow_util.set_conv_layer(data, residual_b, residual_W, strides=strides)
            residual_bn = tensorflow_util.set_batch_normalization(residual_conv, name=name+'_residual_bn')
        else:
            residual_bn = data

        depth_W1 = tensorflow_util.get_init_weight([3, 3, data.shape[3], 1], name=name+'_depth_W1')
        point_W1 = tensorflow_util.get_init_weight([1, 1, 1 * data.shape[3], channels], name=name+'_point_W1')
        b1 = tensorflow_util.get_init_bias([channels], name=name+"_b1")
        conv1 = tensorflow_util.set_depthwise_separable_conv_layer(data, b1, depth_W1, point_W1, strides=1, rate=rate)
        bn1 = tensorflow_util.set_batch_normalization(conv1, name=name+'_bn1')
        relu1 = tensorflow_util.set_relu(bn1, name=name+'_relu1')

        depth_W2 = tensorflow_util.get_init_weight([3, 3, relu1.shape[3], 1], name=name+'_depth_W2')
        point_W2 = tensorflow_util.get_init_weight([1, 1, 1 * relu1.shape[3], channels], name=name+'_point_W2')
        b2 = tensorflow_util.get_init_bias([channels], name=name+"_b2")
        conv2 = tensorflow_util.set_depthwise_separable_conv_layer(relu1, b2, depth_W2, point_W2, strides=1, rate=rate)
        bn2 = tensorflow_util.set_batch_normalization(conv2, name=name+'_bn2')
        relu2 = tensorflow_util.set_relu(bn2, name=name + '_relu2')

        depth_W3 = tensorflow_util.get_init_weight([3, 3, relu2.shape[3], 1], name=name+'_depth_W3')
        point_W3 = tensorflow_util.get_init_weight([1, 1, 1 * relu2.shape[3], channels], name=name+'_point_W3')
        b3 = tensorflow_util.get_init_bias([channels], name=name+"_b3")
        conv3 = tensorflow_util.set_depthwise_separable_conv_layer(relu2, b3, depth_W3, point_W3, strides=strides, rate=rate)
        bn3 = tensorflow_util.set_batch_normalization(conv3, name=name+'_bn3')

        skip = tensorflow_util.set_add(residual_bn, bn3, name=name+'_add')
        output = tensorflow_util.set_relu(skip, name=name+'_output')
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

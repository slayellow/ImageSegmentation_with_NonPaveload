import numpy as np
import tensorflow as tf

from . import npy_loader
from . import tensorflow_util


# Model SegNet 클래스
# Model 구현과 Train에 관련된 Loss, Optimizer 구현
class ModelSegNet:


    def __init__(self, npy_path=None):
        self.vgg_mean = [103.939, 116.779, 123.68]

        self.npy = npy_loader.NPYLoader()
        self.npy.load_npy(npy_path)

    def build_model(self, image, classes, keep_prob):
        print("Build SegNet Model")
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
        bgr = tf.concat(axis=3, values=[blue - self.vgg_mean[0], green - self.vgg_mean[1], red - self.vgg_mean[2]])

        # Layer 1 ( conv3 -> conv3 -> maxpool )
        conv1_1 = self.make_conv_layer(bgr, "conv1_1")
        conv1_2 = self.make_conv_layer(conv1_1, "conv1_2")
        pool1, pool1_indices, shape1 = tensorflow_util.set_max_pool_with_indices(conv1_2)

        # Layer 2 ( conv3 -> conv3 -> maxpool )
        conv2_1 = self.make_conv_layer(pool1, "conv2_1")
        conv2_2 = self.make_conv_layer(conv2_1, "conv2_2")
        pool2, pool2_indices, shape2 = tensorflow_util.set_max_pool_with_indices(conv2_2)

        # Layer 3 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv3_1 = self.make_conv_layer(pool2, "conv3_1")
        conv3_2 = self.make_conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.make_conv_layer(conv3_2, "conv3_3")
        pool3, pool3_indices, shape3 = tensorflow_util.set_max_pool_with_indices(conv3_3)

        # Layer 4 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv4_1 = self.make_conv_layer(pool3, "conv4_1")
        conv4_2 = self.make_conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.make_conv_layer(conv4_2, "conv4_3")
        pool4, pool4_indices, shape4 = tensorflow_util.set_max_pool_with_indices(conv4_3)

        # Layer 5 ( conv3 -> conv3 -> conv3 -> maxpool )
        conv5_1 = self.make_conv_layer(pool4, "conv5_1")
        conv5_2 = self.make_conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.make_conv_layer(conv5_2, "conv5_3")
        pool5, pool5_indices, shape5 = tensorflow_util.set_max_pool_with_indices(conv5_3)

        # Decoder - Upscale to Actual Image ize
        # Encoder와 같은 구조로 Decoder 진행
        # Layer 5 ( unpool -> conv3 -> conv3 -> conv3 )
        unpool_5 = self.make_upsampling(pool5, pool5_indices, shape5, name="unpool_5")

        W_deconv5_1 = tensorflow_util.get_init_weight([3, 3, shape5[3], shape5[3]], name="W_deconv5_1")
        b_deconv5_1 = tensorflow_util.get_init_bias([shape5[3]], name="b_deconv5_1")
        deconv5_1 = tensorflow_util.set_conv_layer(unpool_5, b_deconv5_1, W_deconv5_1)
        bn_deconv5_1 = tensorflow_util.set_batch_normalization(deconv5_1, name='bn_deconv5_1')
        relu_deconv5_1 = tensorflow_util.set_relu(bn_deconv5_1)

        W_deconv5_2 = tensorflow_util.get_init_weight([3, 3, shape5[3], shape5[3]], name="W_deconv5_2")
        b_deconv5_2 = tensorflow_util.get_init_bias([shape5[3]], name="b_deconv5_2")
        deconv5_2 = tensorflow_util.set_conv_layer(relu_deconv5_1, b_deconv5_2, W_deconv5_2)
        bn_deconv5_2 = tensorflow_util.set_batch_normalization(deconv5_2, name='bn_deconv5_2')
        relu_deconv5_2 = tensorflow_util.set_relu(bn_deconv5_2)

        W_deconv5_3 = tensorflow_util.get_init_weight([3, 3, shape5[3], shape5[3]], name="W_deconv5_3")
        b_deconv5_3 = tensorflow_util.get_init_bias([shape5[3]], name="b_deconv5_3")
        deconv5_3 = tensorflow_util.set_conv_layer(relu_deconv5_2, b_deconv5_3, W_deconv5_3)
        bn_deconv5_3 = tensorflow_util.set_batch_normalization(deconv5_3, name='bn_deconv5_3')
        relu_deconv5_3 = tensorflow_util.set_relu(bn_deconv5_3)

        # Layer 4 ( unpool -> conv3 -> conv3 -> conv3 )
        unpool_4 = self.make_upsampling(relu_deconv5_3, pool4_indices, shape4, name="unpool_4")

        W_deconv4_1 = tensorflow_util.get_init_weight([3, 3, shape4[3], shape4[3]], name="W_deconv4_1")
        b_deconv4_1 = tensorflow_util.get_init_bias([shape4[3]], name="b_deconv4_1")
        deconv4_1 = tensorflow_util.set_conv_layer(unpool_4, b_deconv4_1, W_deconv4_1)
        bn_deconv4_1 = tensorflow_util.set_batch_normalization(deconv4_1, name='bn_deconv4_1')
        relu_deconv4_1 = tensorflow_util.set_relu(bn_deconv4_1)

        W_deconv4_2 = tensorflow_util.get_init_weight([3, 3, shape4[3], shape4[3]], name="W_deconv4_2")
        b_deconv4_2 = tensorflow_util.get_init_bias([shape4[3]], name="b_deconv4_2")
        deconv4_2 = tensorflow_util.set_conv_layer(relu_deconv4_1, b_deconv4_2, W_deconv4_2)
        bn_deconv4_2 = tensorflow_util.set_batch_normalization(deconv4_2, name='bn_deconv4_2')
        relu_deconv4_2 = tensorflow_util.set_relu(bn_deconv4_2)

        W_deconv4_3 = tensorflow_util.get_init_weight([3, 3, shape4[3], shape3[3]], name="W_deconv4_3")
        b_deconv4_3 = tensorflow_util.get_init_bias([shape3[3]], name="b_deconv4_3")
        deconv4_3 = tensorflow_util.set_conv_layer(relu_deconv4_2, b_deconv4_3, W_deconv4_3)
        bn_deconv4_3 = tensorflow_util.set_batch_normalization(deconv4_3, name='bn_deconv4_3')
        relu_deconv4_3 = tensorflow_util.set_relu(bn_deconv4_3)

        # Layer 3 ( unpool -> conv3 -> conv3 -> conv3 )
        unpool_3 = self.make_upsampling(relu_deconv4_3, pool3_indices, shape3, name="unpool_3")

        W_deconv3_1 = tensorflow_util.get_init_weight([3, 3, shape3[3], shape3[3]], name="W_deconv3_1")
        b_deconv3_1 = tensorflow_util.get_init_bias([shape3[3]], name="b_deconv3_1")
        deconv3_1 = tensorflow_util.set_conv_layer(unpool_3, b_deconv3_1, W_deconv3_1)
        bn_deconv3_1 = tensorflow_util.set_batch_normalization(deconv3_1, name='bn_deconv3_1')
        relu_deconv3_1 = tensorflow_util.set_relu(bn_deconv3_1)

        W_deconv3_2 = tensorflow_util.get_init_weight([3, 3, shape3[3], shape3[3]], name="W_deconv3_2")
        b_deconv3_2 = tensorflow_util.get_init_bias([shape3[3]], name="b_deconv3_2")
        deconv3_2 = tensorflow_util.set_conv_layer(relu_deconv3_1, b_deconv3_2, W_deconv3_2)
        bn_deconv3_2 = tensorflow_util.set_batch_normalization(deconv3_2, name='bn_deconv3_2')
        relu_deconv3_2 = tensorflow_util.set_relu(bn_deconv3_2)

        W_deconv3_3 = tensorflow_util.get_init_weight([3, 3, shape3[3], shape2[3]], name="W_deconv3_3")
        b_deconv3_3 = tensorflow_util.get_init_bias([shape2[3]], name="b_deconv3_3")
        deconv3_3 = tensorflow_util.set_conv_layer(relu_deconv3_2, b_deconv3_3, W_deconv3_3)
        bn_deconv3_3 = tensorflow_util.set_batch_normalization(deconv3_3, name='bn_deconv3_3')
        relu_deconv3_3 = tensorflow_util.set_relu(bn_deconv3_3)

        # Layer2 ( unpool -> conv3 -> conv3 )
        unpool_2 = self.make_upsampling(relu_deconv3_3, pool2_indices, shape2, name="unpool_2")

        W_deconv2_1 = tensorflow_util.get_init_weight([3, 3, shape2[3], shape2[3]], name="W_deconv2_1")
        b_deconv2_1 = tensorflow_util.get_init_bias([shape2[3]], name="b_deconv2_1")
        deconv2_1 = tensorflow_util.set_conv_layer(unpool_2, b_deconv2_1, W_deconv2_1)
        bn_deconv2_1 = tensorflow_util.set_batch_normalization(deconv2_1, name='bn_deconv2_1')
        relu_deconv2_1 = tensorflow_util.set_relu(bn_deconv2_1)

        W_deconv2_2 = tensorflow_util.get_init_weight([3, 3, shape2[3], shape1[3]], name="W_deconv2_2")
        b_deconv2_2 = tensorflow_util.get_init_bias([shape1[3]], name="b_deconv2_2")
        deconv2_2 = tensorflow_util.set_conv_layer(relu_deconv2_1, b_deconv2_2, W_deconv2_2)
        bn_deconv2_2 = tensorflow_util.set_batch_normalization(deconv2_2, name='bn_deconv2_2')
        relu_deconv2_2 = tensorflow_util.set_relu(bn_deconv2_2)


        # Layer1 ( unpool -> conv3 -> conv3 )
        unpool_1= self.make_upsampling(relu_deconv2_2, pool1_indices, shape1, name="unpool_1")

        W_deconv1_1 = tensorflow_util.get_init_weight([3, 3, shape1[3], shape1[3]], name="W_deconv1_1")
        b_deconv1_1 = tensorflow_util.get_init_bias([shape1[3]], name="b_deconv1_1")
        deconv1_1 = tensorflow_util.set_conv_layer(unpool_1, b_deconv1_1, W_deconv1_1)
        bn_deconv1_1 = tensorflow_util.set_batch_normalization(deconv1_1, name='bn_deconv1_1')
        relu_deconv1_1 = tensorflow_util.set_relu(bn_deconv1_1)

        W_deconv1_2 = tensorflow_util.get_init_weight([3, 3, shape1[3], shape1[3]], name="W_deconv1_2")
        b_deconv1_2 = tensorflow_util.get_init_bias([shape1[3]], name="b_deconv1_2")
        deconv1_2 = tensorflow_util.set_conv_layer(relu_deconv1_1, b_deconv1_2, W_deconv1_2)
        bn_deconv1_2 = tensorflow_util.set_batch_normalization(deconv1_2, name='bn_deconv1_2')
        relu_deconv1_2 = tensorflow_util.set_relu(bn_deconv1_2)

        W_deconv1_3 = tensorflow_util.get_init_weight([1, 1, shape1[3], classes], name="W_deconv1_3")
        b_deconv1_3 = tensorflow_util.get_init_bias([classes], name="b_deconv1_3")
        deconv1_3 = tensorflow_util.set_conv_layer(relu_deconv1_2, b_deconv1_3, W_deconv1_3)

        print("SegNet Model Build Success!")
        return deconv1_3

    def optimize_model(self, last_layer, correct_label, learning_rate, num_classes):
        logits = tf.reshape(last_layer, (-1, num_classes), name="segnet_logits")
        correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label_reshaped[:], logits=logits, name="Loss")
        loss_op = tf.reduce_mean(cross_entropy, name="segnet_loss")

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="segnet_train_op")

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return logits, train_op, loss_op, accuracy

    # conv -> BN -> relu
    def make_conv_layer(self, data, name):
        with tf.compat.v1.variable_scope(name):
            filt = tensorflow_util.get_conv_filter(self.npy.get_values(name, 0), name)
            bias = tensorflow_util.get_bias(self.npy.get_values(name, 1), name)

            conv = tensorflow_util.set_conv_layer(data, bias, filt)
            bn = tensorflow_util.set_batch_normalization(conv, 'bn_'+name)
            relu = tensorflow_util.set_relu(bn)

            return relu

    def make_upsampling(self, pool, indices, output_shape, name=None):
        with tf.compat.v1.variable_scope(name):
            pool_ = tf.reshape(pool, [-1])
            batch = tf.range(tf.shape(pool)[0])
            batch = tf.cast(batch, dtype=indices.dtype)
            batch_range = tf.reshape(batch, [tf.shape(pool)[0], 1, 1, 1])
            b = tf.ones_like(indices) * batch_range
            b = tf.reshape(b, [-1, 1])
            ind_ = tf.reshape(indices, [-1, 1])
            ind_ = tf.concat([b, ind_], 1)
            ret = tf.scatter_nd(ind_, pool_, shape=[tf.shape(pool)[0], output_shape[1] * output_shape[2] * output_shape[3]])
            ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
            return ret
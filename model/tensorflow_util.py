import tensorflow as tf

version = tf.__version__.split('.')


def set_max_pool(data, kernel=2, stride=2, padding='SAME', name='pool'):
    return tf.nn.max_pool2d(data, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding,
                            name=name)


def set_avg_pool(data, kernel=2, stride=2, padding='SAME', name='pool'):
    return tf.nn.avg_pool2d(data, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding, name=name)


def set_max_pool_with_indices(data, kernel=2, stride=2, padding='SAME', name='pool'):
    pool, indices = tf.nn.max_pool_with_argmax(data, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1],
                                               padding=padding, name=name)
    return pool, indices, data.get_shape().as_list()


def set_conv_layer(data, bias, filter, strides=1, padding='SAME'):
    conv = tf.nn.conv2d(data, filter, [1, strides, strides, 1], padding=padding)
    conv_bias = tf.nn.bias_add(conv, bias)
    return conv_bias


def set_atros_conv_layer(data, bias, filter, rate, padding='SAME'):
    conv = tf.nn.atrous_conv2d(data, filter, rate=rate, padding=padding)
    conv_bias = tf.nn.bias_add(conv, bias)
    return conv_bias


def set_relu(data, name=None):
    if name is None:
        return tf.nn.relu(data)
    else:
        return tf.nn.relu(data, name=name)


def set_batch_normalization(data, name=None):
    if name is None:
        return tf.compat.v1.layers.batch_normalization(data)
    else:
        return tf.compat.v1.layers.batch_normalization(data, name=name)


def set_fc_layer(data, weight, bias):
    shape = data.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(data, [-1, dim])
    fc = tf.nn.bias_add(tf.matmul(x, weight), bias)
    return fc


def set_unpool(data, indices, output_shape, name=None):
    with tf.compat.v1.variable_scope(name):
        pool_ = tf.reshape(data, [-1])
        batch = tf.range(tf.shape(data)[0])
        batch = tf.cast(batch, dtype=indices.dtype)
        batch_range = tf.reshape(batch, [tf.shape(data)[0], 1, 1, 1])
        b = tf.ones_like(indices) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(indices, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[tf.shape(data)[0], output_shape[1] * output_shape[2] * output_shape[3]])
        ret = tf.reshape(ret, [tf.shape(data)[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret


def set_dropout(data, keep_prob):
    return tf.nn.dropout(data, rate=1-keep_prob)


def set_concat(data, axis=-1, name='concat'):
    return tf.concat(values=data, axis=axis, name=name)


def set_transpose_conv(data, bias, filter, output_shape, stride=2, padding="SAME"):
    conv = tf.nn.conv2d_transpose(data, filter, output_shape, strides=[1, stride, stride, 1], padding=padding)
    conv_bias = tf.nn.bias_add(conv, bias)
    return conv_bias


def set_add(data1, data2, name=None):
    if name is None:
        return tf.add(data1, data2)
    else:
        return tf.add(data1, data2, name=name)


def set_add_list(data, name=None):
    if name is None:
        return tf.add_n(data)
    else:
        return tf.add_n(data, name=name)


# 전이 학습 데이터 ( filter, bias, weight )를 반환하는 함수
def get_conv_filter(data, name):
    return tf.Variable(data, name='filter_' + name)


def get_bias(data, name):
    return tf.Variable(data, name='biases_' + name)


def get_fcl_weight(data, name):
    return tf.Variable(data, name='weights_' + name)


# 초기 데이터 ( weight, bias ) 를 반환하는 함수
def get_init_weight(data, stddev=0.02, name=None):
    initial = tf.random.truncated_normal(data, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.compat.v1.get_variable(name, initializer=initial)


def get_init_bias(data, name=None):
    initial = tf.constant(0.0, shape=data)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.compat.v1.get_variable(name, initializer=initial)


def get_mIoU(data, label, classes):
    return tf.compat.v1.metrics.mean_iou(label, data, classes, name='mean_iou')


def set_global_average_pooling(data, name=None, keep_dim=True):
    if name is None:
        return tf.reduce_mean(data, [1,2], keep_dims=keep_dim)
    else:
        return tf.reduce_mean(data, [1,2], name=name, keep_dims=keep_dim)


def set_bilinear_upsampling(data, output_shape):
    return tf.compat.v1.image.resize_bilinear(data, output_shape)


def set_depthwise_separable_conv_layer(data, bias, depthwise, pointwise, strides=1, rate=1, padding='SAME'):
    if version[0] is '1':
        conv = tf.nn.separable_conv2d(data, depthwise_filter=depthwise, pointwise_filter=pointwise,
                                  strides=[1, strides, strides, 1], padding=padding, rate=[rate, rate])
    elif version[0] is '2':
        conv = tf.nn.separable_conv2d(data, depthwise_filter=depthwise, pointwise_filter=pointwise,
                                  strides=[1, strides, strides, 1], padding=padding, dilations=rate)
    else:
        print('Too Upper Version')

    conv_bias = tf.nn.bias_add(conv, bias)
    return conv_bias




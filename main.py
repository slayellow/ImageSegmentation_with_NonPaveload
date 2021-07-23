# --------------CPU Only------------------
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# --------------CPU Only------------------
import timeit
from Data.data_loader import DataLoader
from model.fcn import ModelFCN
from model.segnet import ModelSegNet
from model.deeplab_v1 import ModelDeepLab_v1
from model.deeplab_v2 import ModelDeepLab_v2
from model.deeplab_v3 import ModelDeepLab_v3
from model.deeplab_v3_plus import ModelDeepLab_v3_Plus
import util.util_func as uc
from model.unet import ModelUNet
import tensorflow as tf
import tkinter
from tkinter import filedialog

if not tf.test.gpu_device_name():
    print("No GPU Found!")
else:
    print("Default GPU Device : {}".format(tf.test.gpu_device_name()))


def main(argv=None):
    logs_dir = 'logs/DEEPLAB_V3_PLUS/'
    num_class = 2
    max_iteration = 50430
    batch_size = 2

    # Model Size Variable
    image_shape = (480, 640)

    root = tkinter.Tk()
    root.withdraw()
    img_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Input Image Dir')
    label_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Label Image Dir')

    tf.compat.v1.reset_default_graph()

    # Dropout
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")

    # Image & Correct Label
    # ---------------- 1. General Case ----------------
    # image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    # correct_label = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="correct_label")

    # ---------------- 2. You Know Image Shape ----------------
    image = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, image_shape[0], image_shape[1], 3],
                                     name="input_image")

    # Correct Label : 2
    correct_label = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, image_shape[0], image_shape[1], 2],
                                             name="correct_label")

    # ------------------ DeepLab_v1 Model --------------------
    # model = ModelDeepLab_v1(npy_path='./model/vgg16.npy')
    # layer = model.build_model(image, classes=num_class, keep_prob=keep_prob)

    # ------------------- DeepLab_v2 Model ------------------------
    # model = ModelDeepLab_v2()
    # layer = model.build_model(image, classes=num_class, first=64)

    # ------------------ DeepLab_V3 Model ----------------------------
    # model = ModelDeepLab_v3(block_list=[3, 4, 23, 3])
    # layer = model.build_model(image, classes=num_class, first=256)

    # ----------------- DeepLab_V3+ Model -------------------
    model = ModelDeepLab_v3_Plus()
    layer = model.build_model(image, classes=num_class, first=16, flow=16)
    _, train_op, loss_op, accuracy, mIoU, update_op = model.optimize_model(layer, correct_label, learning_rate=1e-5,
                                                                           num_classes=num_class)

    tf.compat.v1.summary.scalar("loss_op", loss_op)
    tf.compat.v1.summary.scalar("accuracy", accuracy)
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter(logs_dir)

    # Get Total Parameter
    total_parameter = 0
    print("--------------------")
    print(" Model Parameter Size ")
    for variable in tf.compat.v1.trainable_variables():
        print("--------------------")
        shape = variable.get_shape()
        print("DType : %s" % str(variable.dtype))
        print("Shape : %s" % str(shape))
        print("Shape Length : %s" % str(len(shape)))
        variable_parameter = 1
        for dim in shape:
            print("Dimension : %s" % str(dim))
            variable_parameter *= dim
        print("Variable Parameter : %s" % str(variable_parameter))
        total_parameter += variable_parameter
    print("Total Paramters : %s" % str(total_parameter))

    train_data_loader = DataLoader(img_path, label_path, batch_size=batch_size)

    ### GPU Setting
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # --------------CPU Only------------------
    # sess = tf.compat.v1.Session()
    # --------------CPU Only------------------

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model Restored")

    for itr in range(max_iteration):
        images, labels = train_data_loader.get_data_label()
        feed_dict = {image: images, correct_label: labels, keep_prob: 0.5}

        merge, _ = sess.run([merged, train_op], feed_dict=feed_dict)
        writer.add_summary(merge, itr)

        sess.run(update_op, feed_dict=feed_dict)
        miou = sess.run(mIoU, feed_dict=feed_dict)

        if itr % 1000 == 0 and itr > 0:
            print("Saving Model to file in %s" % logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr)

        if itr % 100 == 0:
            feed_dict = {image: images, correct_label: labels, keep_prob: 1}
            total_loss = sess.run(loss_op, feed_dict=feed_dict)
            cur_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("Step %s, Train Loss = %s ,Accuracy : %s, mIoU : %s ||  Memory Use : %s GB" % (
                str(itr), str(total_loss), str(cur_accuracy), str(miou), str(uc.memory())))


start_time = timeit.default_timer()
main()
print("Finished")
terminate_time = timeit.default_timer()
print("Program Execute Time : %f Second" % (terminate_time - start_time))

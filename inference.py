from Data.data_loader import DataLoader
from model.fcn import ModelFCN
from model.deeplab_v1 import ModelDeepLab_v1
from model.deeplab_v2 import ModelDeepLab_v2
from model.deeplab_v3 import ModelDeepLab_v3
from model.deeplab_v3_plus import ModelDeepLab_v3_Plus
from model.unet import ModelUNet
import tensorflow as tf
import tkinter
from tkinter import filedialog
import util.util_func as uc
import scipy.misc as misc
import numpy as np
import timeit
import os


def main(argv=None):
    logs_dir = 'logs/DEEPLAB_V3_PLUS'
    output_dir = 'output/DEEPLAB_V3_PLUS/'
    num_class = 2
    max_iteration = 1000

    root = tkinter.Tk()
    root.withdraw()
    img_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Test Input Image Dir')

    tf.compat.v1.reset_default_graph()

    # Dropout
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")

    ### DeepLab_v3
    model = ModelDeepLab_v3_Plus()
    layer = model.build_model(image, classes=num_class, first=16, flow=16)
    logits = tf.compat.v1.reshape(layer, (-1, num_class), name="deeplab_v3_plus_logits")
    softmax = tf.nn.softmax(logits)

    test_data_loader = DataLoader(img_path, "", batch_size=1, shuffle=False)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.inter_op_parallelism_threads = 1
    # config.intra_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(config=config)

    print("Setting up Saver")
    saver = tf.compat.v1.train.Saver()

    sess.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.compat.v1.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored... " + str(ckpt.model_checkpoint_path))
    else:
        print("No Train File! Train First!!")
        return

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(output_dir + "/OverLay"): os.makedirs(output_dir + "/OverLay")

    cnt = 0
    print("Start Predicting " + str(test_data_loader.num_files) + " images")

    while test_data_loader.itr < test_data_loader.num_files:
        start_time = timeit.default_timer()

        images, file_name = test_data_loader.get_data()

        im_softmax = sess.run(softmax, {keep_prob: 1.0, image: images})

        terminate_time = timeit.default_timer()
        print("File : %s || Time : %f Sec || Memory : %s || %s %%" % (file_name[:-4], terminate_time - start_time,
                                                                      str(uc.memory()),
                                                                      str(cnt * 100.0 / test_data_loader.num_files)))
        # road
        im_softmax = im_softmax[:, 1].reshape(images.shape[1], images.shape[2])
        segmentation1 = (im_softmax > 0.5).reshape(images.shape[1], images.shape[2], 1)
        mask1 = np.dot(segmentation1, np.array([[255, 0, 0, 80]]))
        mask1 = misc.toimage(mask1, mode="RGBA")

        street_im = misc.toimage(images[0])
        street_im.paste(mask1, box=None, mask=mask1)
        # street_im.paste(mask2, box=None, mask=mask2)
        # street_im.paste(mask3, box=None, mask=mask3)

        misc.imsave(output_dir + "/OverLay/" + file_name[:-4] + ".png", np.array(street_im))


        cnt += 1
        tf.contrib.keras.backend.clear_session()


main()
print("Finished")

# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import click
import shutil
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.image import resize_image


# tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', '', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    if FLAGS.image:
        files.append(FLAGS.image)
    else:
        for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
    print('Find {} images'.format(len(files)))
    return files


def main(argv=None):

    print('Mode :%s' % FLAGS.detect_mode)

    sys.path.append(os.getcwd())

    from utils.text_connector.detectors import TextDetector
    from nets import model_train as model
    from utils.rpn_msr.proposal_layer import proposal_layer

    if FLAGS.output_path:
        if os.path.exists(FLAGS.output_path):
            shutil.rmtree(FLAGS.output_path)
        os.makedirs(FLAGS.output_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.compat.v1.get_default_graph().as_default():
        input_image = tf.compat.v1.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.compat.v1.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.compat.v1.train.Saver(
            variable_averages.variables_to_restore())

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(
                ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)

                try:
                    im = cv2.imread(im_fn)  # [:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, im_scale = resize_image(im, int(FLAGS.image_size))
                img = cv2.detailEnhance(img)

                # process image
                start = time.time()
                h, w, c = img.shape
                print(h, w, im_scale)
                im_info = np.array([h, w, c]).reshape([1, 3])

                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                thickness = int(h / 100)
                textsegs, _ = proposal_layer(
                    cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE=FLAGS.detect_mode)
                boxes = textdetector.detect(
                    textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.float64)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                # applied to result and fix scale
                for i, box in enumerate(boxes):
                    box[:8] /= im_scale
                    points = [box[:8].astype(np.int32).reshape((-1, 1, 2))]
                    cv2.polylines(im, points, True, color=(0, 255, 0),
                                  thickness=thickness, lineType=cv2.LINE_AA)

                basename = os.path.basename(im_fn)
                if FLAGS.output_path:
                    cv2.imwrite(os.path.join(FLAGS.output_path,
                                             basename), im)

                    with open(os.path.join(FLAGS.output_path, os.path.splitext(basename)[0]) + ".txt",
                              "w") as f:
                        for i, box in enumerate(boxes):
                            line = ",".join(str(box[k]) for k in range(8))
                            line += "," + str(scores[i]) + "\r\n"
                            f.writelines(line)
                else:
                    # cv2.namedWindow(basename, cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty(
                    #     basename, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(basename, w, h)
                    cv2.imshow(basename, im)
                    cv2.waitKey(0)


@click.command()
@click.option('--mode', '-m', default='H')
@click.option('--image', '-i', default='data/demo/cmt.jpg')
@click.option('--size', '-s', default='600')
def run(mode, image, size):
    tf.app.flags.DEFINE_string('detect_mode', mode, '')
    tf.app.flags.DEFINE_string('image', image, '')
    tf.app.flags.DEFINE_string('image_size', size, '')

    tf.compat.v1.app.run()


if __name__ == '__main__':
    run()

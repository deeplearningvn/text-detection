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

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
FLAGS = tf.app.flags.FLAGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


exts = ['.jpg', '.png', '.jpeg', '.JPG']


def get_images():
    files = []
    _, ext = os.path.splitext(FLAGS.input)
    if ext in exts:
        files.append(FLAGS.input)
    else:
        for parent, dirnames, filenames in os.walk(FLAGS.input):
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
        # if need overide output? may be no need for testing
        # shutil.rmtree(FLAGS.output_path)

        if not os.path.exists(FLAGS.output_path):
            os.makedirs(FLAGS.output_path)

        image_path = os.path.join(FLAGS.output_path, "image")
        label_path = os.path.join(FLAGS.output_path, "label")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

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
            FLAGS.moving_average_decay, global_step)
        saver = tf.compat.v1.train.Saver(
            variable_averages.variables_to_restore())

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(
                ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            # print(im_fn_list)

            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)

                try:
                    im = cv2.imread(im_fn)  # [:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im, FLAGS.image_size)
                img = cv2.detailEnhance(img)

                # process image
                start = time.time()
                h, w, c = img.shape
                # print(h, w, rh, rw)
                im_info = np.array([h, w, c]).reshape([1, 3])

                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                thickness = max(1, int(im.shape[0] / 400))
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
                    box[:8][::2] /= rh
                    box[1:8][::2] /= rh

                basename = os.path.basename(im_fn)
                if FLAGS.output_path:

                    bfn, ext = os.path.splitext(basename)
                    gt_path = os.path.join(
                        FLAGS.output_path, "label", 'gt_' + bfn + '.txt')
                    img_path = os.path.join(
                        FLAGS.output_path, "image", basename)
                    # save image and coordination, may be resize image
                    # cv2.imwrite(img_path, im)
                    shutil.copyfile(im_fn, img_path)
                    with open(gt_path, "w") as f:
                        for i, box in enumerate(boxes):
                            line = ",".join(str(int(box[k])) for k in range(8))
                            line += "," + str(scores[i]) + "\r\n"
                            f.writelines(line)
                else:
                    # cv2.namedWindow(basename, cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty(
                    #     basename, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                    # draw polyline and show
                    for i, box in enumerate(boxes):
                        points = [box[:8].astype(np.int32).reshape((-1, 1, 2))]
                        cv2.polylines(im, points, True, color=(0, 255, 0),
                                      thickness=thickness, lineType=cv2.LINE_AA)
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(basename, w, h)
                    cv2.imshow(basename, im)
                    cv2.waitKey(0)


@click.command()
@click.option('--mode', '-m', default='O')  # H and O
@click.option('--input', '-i', default='data/demo/cmt.jpg')
@click.option('--size', '-s', default=600, type=int)
@click.option('--gpu', '-g', default='0')
@click.option('--checkpoint_path', '-cp', default='checkpoints_mlt/')
@click.option('--output', '-o', default='')
def run(mode, input, size, gpu, checkpoint_path, output):
    tf.app.flags.DEFINE_string('output_path', output, '')
    tf.app.flags.DEFINE_string('detect_mode', mode, '')
    tf.app.flags.DEFINE_string('input', input, '')
    tf.app.flags.DEFINE_integer('image_size', size, '')
    tf.app.flags.DEFINE_string('gpu', gpu, '')
    tf.app.flags.DEFINE_string('checkpoint_path', checkpoint_path, '')

    tf.compat.v1.app.run()


if __name__ == '__main__':
    run()

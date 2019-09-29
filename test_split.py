import os
import sys
import click
import cv2
import numpy as np
from utils.dataset.data_provider import load_annoataion


@click.command()
@click.option('--input', '-i', default='data/dataset/mlt_cmt')
@click.option('--name', '-n')
def process(input, name):
    im_fn = os.path.join(input, "image", name)
    im = cv2.imread(im_fn)
    h, w, c = im.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    fn, _ = os.path.splitext(name)
    txt_fn = os.path.join(input, "label", fn + '.txt')

    if not os.path.exists(txt_fn):
        print("Ground truth for image {} not exist!".format(im_fn))
        return
    bbox = load_annoataion(txt_fn)
    if len(bbox) == 0:
        print("Ground truth for image {} empty!".format(im_fn))
        return

    for p in bbox:
        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(
            0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow(name, im)
    cv2.waitKey(0)


if __name__ == '__main__':
    process()

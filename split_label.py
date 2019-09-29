import os
import sys
import click
import cv2
import numpy as np
from tqdm import tqdm

from utils.image import resize_image

sys.path.append(os.getcwd())
from utils.prepare.utils import orderConvex, shrink_poly


@click.command()
@click.option('--input', '-i', default='mlt_selected')
@click.option('--output', '-o', default='data/dataset/mlt_cmt')
@click.option('--size', '-s', default='600', type=int)
def process(input, output, size):

    im_fns = os.listdir(os.path.join(input, "image"))
    im_fns.sort()

    if not os.path.exists(os.path.join(output, "image")):
        os.makedirs(os.path.join(output, "image"))
    if not os.path.exists(os.path.join(output, "label")):
        os.makedirs(os.path.join(output, "label"))

    for im_fn in tqdm(im_fns):
        try:
            _, fn = os.path.split(im_fn)
            bfn, ext = os.path.splitext(fn)
            if ext.lower() not in ['.jpg', '.png']:
                continue

            gt_path = os.path.join(input, "label", 'gt_' + bfn + '.txt')
            img_path = os.path.join(input, "image", im_fn)

            img = cv2.imread(img_path)
            h, w, _ = img.shape
            re_im, im_scale = resize_image(img, size)
            re_size = re_im.shape

            polys = []
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                splitted_line = line.strip().lower().split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
                poly = np.array(
                    [x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
                poly[:, 0] = poly[:, 0] / w * re_size[1]
                poly[:, 1] = poly[:, 1] / h * re_size[0]
                poly = orderConvex(poly)
                polys.append(poly)

                # cv2.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

            res_polys = []
            for poly in polys:
                # delete polys with width less than 10 pixel
                if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                    continue

                res = shrink_poly(poly)
                # for p in res:
                #     cv2.polylines(re_im, [p.astype(np.int32).reshape(
                #         (-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

                res = res.reshape([-1, 4, 2])
                for r in res:
                    x_min = np.min(r[:, 0])
                    y_min = np.min(r[:, 1])
                    x_max = np.max(r[:, 0])
                    y_max = np.max(r[:, 1])

                    res_polys.append([x_min, y_min, x_max, y_max])

            cv2.imwrite(os.path.join(output, "image", fn), re_im)
            with open(os.path.join(output, "label", bfn) + ".txt", "w") as f:
                for p in res_polys:
                    line = ",".join(str(p[i]) for i in range(4))
                    f.writelines(line + "\r\n")
                    # for p in res_polys:
                    #     cv2.rectangle(re_im, (p[0], p[1]), (p[2], p[3]), color=(
                    #         0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                    # cv2.imshow("demo", re_im)
                    # cv2.waitKey(0)
        except:
            print("Error processing {}".format(im_fn))


if __name__ == '__main__':
    process()

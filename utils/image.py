import cv2


def resize_image(img, max_size=600):
    h, w, _ = img.shape
    im_size_max = max(h, w)

    im_scale = float(max_size) / float(im_size_max)

    # new_h = int(h * im_scale)
    # new_w = int(w * im_scale)

    # # round to 16
    # new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    # new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, None, fx=im_scale, fy=im_scale,
                       interpolation=cv2.INTER_LINEAR)
    return re_im, im_scale

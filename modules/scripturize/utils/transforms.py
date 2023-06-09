import numpy as np
import cv2


def crop_foreground(img, axis=None):
    xs = np.max(img, axis=0)
    ys = np.max(img, axis=1)
    valx = np.where(xs)
    valy = np.where(ys)
    if not len(valx[0]):
        axis = 1
    elif not len(valy[0]):
        return img
    if axis is None:
        img = crop_foreground(img, axis=0)
        img = crop_foreground(img, axis=1)
    elif axis == 0:
        valx = valx[0]
        if not len(valx):
            return img
        x1, x2 = valx[[0, -1]]
        img = img[:, x1:x2 + 1]
    elif axis == 1:
        valy = valy[0]
        if not len(valy):
            return img
        y1, y2 = valy[[0, -1]]
        img = img[y1:y2 + 1, :]
    return img


def match_prototype(image, template):
    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
    max_response = np.max(response)
    locations = np.where(response >= min(max_response, 0.7))
    return max_loc, locations


def center_text(manuscript, padding=20):
    manuscript = crop_foreground(manuscript)
    manuscript = cv2.copyMakeBorder(manuscript, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=0)
    return manuscript

import numpy as np
import cv2 
class Utils:
    def __init__(self) -> None:
        pass

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32, img_size=(640, 640)):
        shape = im.shape[:2]
        new_shape = img_size
        if isinstance(new_shape, int):
            new_shape=(new_shape, new_shape)
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        new_unpad = int(round(shape[1]* r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw/=2
        dh/=2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

import cv2
import numpy as np
from PIL import Image

'''
Utility to automatically crop bananas out of images (reduces background noise)
'''

def autocrop_banana(in_path: str, out_path: str = None, pad: int = 12) -> Image.Image:

    #read + shrink
    bgr = cv2.imread(in_path)
    if bgr is None:
        raise FileNotFoundError(in_path)
    h, w = bgr.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    #crude foreground
    v = hsv[...,2]/255.0
    s = hsv[...,1]/255.0
    fg = (v < 0.9) | (s > 0.15)

    #prefer yellow/brown hues
    hdeg = hsv[...,0] * 2.0
    yellow = (hdeg >= 25) & (hdeg <= 70) & (s > 0.2)
    brown  = (hdeg >= 5)  & (hdeg <= 45) & (s > 0.2) & (v < 0.7)
    dark   = v < 0.3

    mask = (fg & (yellow | brown | dark)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=2)

    #biggest contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        #fallback. use whole image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if out_path: pil.save(out_path)
        return pil

    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    #unscale bbox to original if it downscaled
    if scale < 1.0:
        inv = 1/scale
        x,y,w,h = [int(round(v*inv)) for v in (x,y,w,h)]
        bgr_full = cv2.imread(in_path)
    else:
        bgr_full = bgr

    #pad + clamp
    H,W = bgr_full.shape[:2]
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
    crop = bgr_full[y0:y1, x0:x1]

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if out_path:
        pil.save(out_path)
    return pil
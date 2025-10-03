from PIL import Image
import numpy as np
from auto_crop import autocrop_banana

'''
Defines color_percents(path) which extracts yellow% and dark% pixels from cropped banana images
'''

def color_percents(image_path: str):
    img = autocrop_banana(image_path)
    img = img.convert("RGB").resize((480, 360))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.max(arr, axis=-1);
    cmin = np.min(arr, axis=-1)
    delta = cmax - cmin + 1e-6
    hue = np.zeros_like(cmax)
    m = delta > 0
    rmax = (cmax == r) & m;
    gmax = (cmax == g) & m;
    bmax = (cmax == b) & m
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    hue *= 60.0
    sat = delta / (cmax + 1e-6);
    val = cmax

    #foreground now = banana
    fg = (sat > 0.12) | (val < 0.9)

    yellow = (hue >= 30) & (hue <= 70) & (sat >= 0.25) & (val >= 0.35)
    brown = (hue >= 5) & (hue <= 45) & (sat >= 0.2) & (val <= 0.7)
    black = (val <= 0.3)
    dark = brown | black

    N = max(fg.sum(), 1)
    y = 100.0 * (yellow & fg).sum() / N
    d = 100.0 * (dark & fg).sum() / N
    return round(float(y), 1), round(float(d), 1)


if __name__ == "__main__":
    print("Original method:", color_percents("../data/raw/banana_set_1/Screenshot_16.jpg"))
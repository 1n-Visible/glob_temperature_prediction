
__all__=[
    "remap", "hsv_to_rgb", "create_region"
]

import numpy as np

_hsv_table = [
    [0, 1, 2], [1, 0, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0], [0, 2, 1]
]

def remap(array, /, new_max) -> np.ndarray:
    return array*new_max/np.max(array)

#@njit(fastmath=True)
def hsv_to_rgb(hue, saturation, value):
    h_ = (3*hue/np.pi)%6
    h = np.abs(h_%2 - 1)
    shape=(hue*saturation*value).shape
    rgb = np.ones((3,)+shape, dtype=np.float32)
    saturation*=rgb[0]
    
    h0 = (0<=h_) & (h_<1)
    h1 = (1<=h_) & (h_<2)
    h2 = (2<=h_) & (h_<3)
    h3 = (3<=h_) & (h_<4)
    h4 = (4<=h_) & (h_<5)
    h5 = (5<=h_) & (h_<6)
    X = (1-h*saturation)
    X0 = 1-saturation
    rgb[0, h1|h4] = X[h1|h4]
    rgb[1, h0|h3] = X[h0|h3]
    rgb[2, h2|h5] = X[h2|h5]
    rgb[0, h2|h3] = X0[h2|h3]
    rgb[1, h4|h5] = X0[h4|h5]
    rgb[2, h0|h1] = X0[h0|h1]
    # [value, value*(1-h*saturation), value*(1-saturation)]
    return rgb*value

def create_region(pos, diameter, size):
    region0 = np.reshape(pos, (2, 1))+np.int32([[-diameter, diameter]]*2)//2
    return tuple(slice(*np.int32(np.clip(r, 0, dim)))
                 for r, dim in zip(region0, size))

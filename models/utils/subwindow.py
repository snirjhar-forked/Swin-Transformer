import numpy as np
import numba as nb
import torch

from .generator import GIndex2DGen

@nb.njit(nogil=True)
def subindex_(coords, window_size, subwindow_size, indices, relatives):
    h, w, _ = coords.shape
    H, W = h//subwindow_size, w//subwindow_size

    stride1 = subwindow_size
    stride2 = subwindow_size * subwindow_size
    stride3 = w * subwindow_size

    max_r = window_size - 1
    range_r = 2*window_size - 1

    indices = indices.reshape(H,W,subwindow_size,subwindow_size)
    relatives = relatives.reshape(H,W,subwindow_size,subwindow_size,
                                      subwindow_size,subwindow_size)
    for Y in range(H):
        for X in range(W):
            Yoff = Y * subwindow_size
            Xoff = X * subwindow_size
            for y in range(subwindow_size):
                for x in range(subwindow_size):
                    yy = Yoff + y
                    xx = Xoff + x
                    
                    cyy = coords[yy,xx,0]
                    cxx = coords[yy,xx,1]
                    cy = cyy % subwindow_size
                    cx = cxx % subwindow_size
                    cY = cyy // subwindow_size
                    cX = cxx // subwindow_size
                    indices[Y,X,y,x] = (cx + cy*stride1
                                        + cX*stride2 + cY*stride3)
                    for y2 in range(subwindow_size):
                        for x2 in range(subwindow_size):
                            yy2 = Yoff + y2
                            xx2 = Xoff + x2
                            relatives[Y,X,y2,x2,y,x] = ((yy2-cyy+max_r)*range_r
                                                          + xx2-cxx+max_r)


class SubindexGen:
    def __init__(self, window_size, subwindow_size, input_resolution, shift_size, std,
                 buffer_size=10000, warmup=200):
        self.window_size = window_size
        self.subwindow_size = subwindow_size
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.std = std
        
        self.indices = torch.zeros(self.input_resolution[0]*self.input_resolution[1],
                                   dtype=torch.long, pin_memory=True)
        self.relatives = torch.zeros(self.input_resolution[0]*self.input_resolution[1]*
                                     self.subwindow_size*self.subwindow_size,
                                     dtype=torch.long, pin_memory=True)
        
        self.randgen = GIndex2DGen(self.input_resolution, self.std,
                                   self.window_size, self.shift_size,
                                   buffer_size, warmup)

    def __iter__(self):
        for coords in self.randgen:
            subindex_(coords, self.window_size, self.subwindow_size,
                      self.indices.numpy(), self.relatives.numpy())
            yield self.indices, self.relatives

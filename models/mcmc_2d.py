import numba as nb
import numpy as np

@nb.njit
def index_2d(dims):
    h, w = dims
    indices = np.empty((h,w,2), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            indices[y,x,0] = y
            indices[y,x,1] = x
    return indices

@nb.njit
def alpha_2d(dims, stds, wins, shifts):
    h,w = dims
    sigma_y, sigma_x = stds
    y_win, x_win = wins
    y_shift, x_shift = shifts

    py = 1/(sigma_y**2)
    px = 1/(sigma_x**2)
    alpha_y = np.zeros((h,h), dtype=np.float64)
    alpha_x = np.zeros((w,w), dtype=np.float64)
    for cy1 in range(h):
        for cy2 in range(h):
            if (cy1-y_shift)//y_win == (cy2-y_shift)//y_win:
                alpha_y[cy1,cy2] = np.exp(py*(cy1-cy2))
    for cx1 in range(w):
        for cx2 in range(w):
            if (cx1-x_shift)//x_win == (cx2-x_shift)//x_win:
                alpha_x[cx1,cx2] = np.exp(px*(cx1-cx2))
    return alpha_y, alpha_x

@nb.njit
def gshuf_2d(indices, alphas, inner_dims, outer_dims, steps):
    h1,w1 = inner_dims
    h2,w2 = outer_dims
    alpha_y, alpha_x = alphas
    for _ in range(steps):
        ind_y1 = indices[0]
        for y2 in range(1,h2):
            ind_y2 = indices[y2]
            w = w2 if y2<h1 else w1
            for x in range(w):
                cy1 = ind_y1[x,0]
                cy2 = ind_y2[x,0]
                cx1 = ind_y1[x,1]
                cx2 = ind_y2[x,1]
                
                alpha = alpha_y[cy1,cy2]
                noise = np.random.rand()
                rf = nb.int64(noise <= alpha)
                ind_y1[x,0] = cy1+rf*(cy2-cy1)
                ind_y1[x,1] = cx1+rf*(cx2-cx1)
                ind_y2[x,0] = cy2+rf*(cy1-cy2)
                ind_y2[x,1] = cx2+rf*(cx1-cx2)
            ind_y1 = ind_y2

        
        ind_x1 = indices[:,0]
        for x2 in range(1,w2):
            ind_x2 = indices[:,x2]
            h = h2 if x2<w1 else h1
            for y in range(h):
                cx1 = ind_x1[y,1]
                cx2 = ind_x2[y,1]
                cy1 = ind_x1[y,0]
                cy2 = ind_x2[y,0]
                
                alpha = alpha_x[cx1,cx2]
                noise = np.random.rand()
                rf = nb.int64(noise <= alpha)
                ind_x1[y,0] = cy1+rf*(cy2-cy1)
                ind_x1[y,1] = cx1+rf*(cx2-cx1)
                ind_x2[y,0] = cy2+rf*(cy1-cy2)
                ind_x2[y,1] = cx2+rf*(cx1-cx2)
            ind_x1 = ind_x2
        
    return indices

@nb.njit
def gindex_2d(dims, stds, wins, shifts, steps):
    indices = index_2d(dims)
    alphas = alpha_2d(dims, stds, wins, shifts)
    indices = gshuf_2d(indices, alphas, dims, dims, steps)
    return indices


@nb.njit
def gindex(dims, std, window_size, shift, steps):
    h, w = dims
    subwindow_size = window_size//2
    H, W = h//subwindow_size, w//subwindow_size

    wins = (window_size, window_size)
    stds = (std, std)
    shifts = (shift, shift)
    coords = gindex_2d(dims, stds, wins, shifts, steps)

    stride1 = subwindow_size
    stride2 = subwindow_size * subwindow_size
    stride3 = w * subwindow_size

    max_r = window_size - 1
    range_r = 2*window_size - 1

    indices = np.empty((H,W,subwindow_size,subwindow_size), dtype=np.int64)
    relatives = np.empty((H,W,subwindow_size,subwindow_size,
                          subwindow_size,subwindow_size),
                         dtype=np.int64)
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
                            relatives[Y,X,y2,x2,y,x] = ((cyy-yy2+max_r)*range_r
                                                          + cxx-xx2+max_r)
    indices = indices.ravel()
    relatives = relatives.ravel()
    return indices, relatives

if __name__ == '__main__':
    shuffled_indices = gindex_2d((16,16), (1,1), 200) #(4,4),
    for i in range(shuffled_indices.shape[0]):
        print('\t'.join([f'({x:2d},{y:2d})' for x,y in shuffled_indices[i]]))
    
    # indices_1d, rev_indices_1d = block_index((16,16), (4,4))
    # shuffled_indices_1d = index_rearrange(shuffled_indices, indices_1d, rev_indices_1d)
    # print(np.arange(16*16,dtype=int).reshape(16,16))
    # print(indices_1d)
    # print(rev_indices_1d)
    # print(rev_indices_1d[indices_1d])
    # print(shuffled_indices_1d)
    # print(shuffled_indices_1d.reshape(4,4,4,4).swapaxes(1,2).reshape(16,16))


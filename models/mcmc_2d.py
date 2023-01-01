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
def alpha_2d(dims, stds):
    h,w = dims
    sigma_y, sigma_x = stds

    py = 1/(sigma_y**2)
    px = 1/(sigma_x**2)
    alpha_y = np.empty((h,h), dtype=np.float64)
    alpha_x = np.empty((w,w), dtype=np.float64)
    for cy1 in range(h):
        for cy2 in range(h):
            alpha_y[cy1,cy2] = np.exp(py*(cy1-cy2))
    for cx1 in range(w):
        for cx2 in range(w):
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
def hgshuf_2d(indices, alphas, num_blocks, steps):
    h, w, _ = indices.shape
    by, bx = num_blocks

    cur_indices = indices.copy()
    for ny in range(by):
        for nx in range(bx):
            top = (ny*h)//by
            bottom = (ny*h+h)//by
            left = (nx*w)//bx
            right = (nx*w+w)//bx
            inner_dims = (top, right)
            outer_dims = (bottom, w)
            cur_indices = gshuf_2d(cur_indices, alphas, inner_dims, outer_dims, steps)
            indices[top:bottom,left:right] = cur_indices[top:bottom,left:right]
    return indices

@nb.njit
def gindex_2d(dims, stds, steps):
    indices = index_2d(dims)
    alphas = alpha_2d(dims, stds)
    indices = gshuf_2d(indices, alphas, dims, dims, steps)
    return indices

@nb.njit
def hgindex_2d(dims, stds, num_blocks, steps):
    indices = index_2d(dims)
    alphas = alpha_2d(dims, stds)
    indices = hgshuf_2d(indices, alphas, num_blocks, steps)
    return indices


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


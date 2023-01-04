import numba as nb
import numpy as np

@nb.njit(nogil=True)
def index_2d(dims):
    h, w = dims
    indices = np.empty((h,w,2), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            indices[y,x,0] = y
            indices[y,x,1] = x
    return indices

@nb.njit(nogil=True)
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

@nb.njit(nogil=True)
def gsample_2d_(index_buffer, alphas, sample_buffer, warmup=0):
    h,w,_ = index_buffer.shape
    alpha_y, alpha_x = alphas
    num_samples = sample_buffer.shape[0]
    
    placement_order = np.arange(num_samples, dtype=np.int64)
    np.random.shuffle(placement_order)
    
    num_steps = warmup + num_samples
    for cur_step in range(num_steps):
        ind_y1 = index_buffer[0]
        for y2 in range(1,h):
            ind_y2 = index_buffer[y2]
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

        
        ind_x1 = index_buffer[:,0]
        for x2 in range(1,w):
            ind_x2 = index_buffer[:,x2]
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
        
        if cur_step >= warmup:
            sample_buffer[placement_order[cur_step-warmup]] = index_buffer

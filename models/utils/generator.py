
import numpy as np
from multiprocessing.pool import ThreadPool

from .mcmc_2d import index_2d, alpha_2d, gsample_2d_

class GIndex2DGen:
    def __init__(self, dims, stds, wins, shifts, buffer_size=10000, warmup=200):
        self.dims = self._to_2tuple(dims)
        self.stds = self._to_2tuple(stds)
        self.wins = self._to_2tuple(wins)
        self.shifts = self._to_2tuple(shifts)
        self.buffer_size = buffer_size
        self.warmup = warmup
        
        self.index_buffer = index_2d(self.dims)
        self.alphas = alpha_2d(self.dims, self.stds, self.wins, self.shifts)
        
        self.sample_buffer = self._make_sample_buffer()
        gsample_2d_(self.index_buffer, self.alphas, self.sample_buffer, self.warmup)
    
    def _to_2tuple(self, x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 2, "Expected 2-tuple"
            return tuple(x)
        return (x, x)
    
    def _make_sample_buffer(self):
        return np.zeros((self.buffer_size,*self.index_buffer.shape),
                        dtype=self.index_buffer.dtype)
    
    def __iter__(self):
        with ThreadPool(processes=1) as pool:
            new_sample_buffer = self._make_sample_buffer()
            task = pool.apply_async(gsample_2d_, (self.index_buffer, self.alphas,
                                                  new_sample_buffer))
                                                  
            while True:
                for sample in self.sample_buffer:
                    yield sample
                task.wait()
                temp = self.sample_buffer
                self.sample_buffer = new_sample_buffer
                new_sample_buffer = temp
                task = pool.apply_async(gsample_2d_, (self.index_buffer, self.alphas,
                                                      new_sample_buffer))
        
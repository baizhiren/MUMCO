import random
import numpy as np

from envs.Entity import Entity
import envs.settings as settings

class DiscreteMap:
    
    def __init__(self, length: int, width: int):
        self.length = length
        self.width = width
        self.map = [[None for _ in range(width)] for _ in range(length)]

    
    def generate(self):
        x = random.randint(0, self.length - 1)
        y = random.randint(0, self.width - 1)
        return (x, y)

    def gaussian_distri_generate(self, mu_x=None, mu_y=None, sigma=1.0):
        
        
        if mu_x is None:
            mu_x = self.length / 2
        if mu_y is None:
            mu_y = self.width / 2
            
        
        x = np.random.normal(mu_x, sigma)
        y = np.random.normal(mu_y, sigma)
        
        
        x = int(np.clip(x, 0, self.length - 1))
        y = int(np.clip(y, 0, self.width - 1))
        
        return (x, y)
    
    def multi_gaussian_generate(self, centers, weights=None, sigma=1.0):
        
        
        if weights is None:
            weights = [1.0] * len(centers)
        
        
        if isinstance(sigma, (list, tuple, np.ndarray)):
            if len(sigma) != len(centers):
                raise ValueError("sigma列表的长度必须与centers列表的长度相同")
            sigma_list = sigma
        else:
            sigma_list = [sigma] * len(centers)
        
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        
        center_idx = np.random.choice(len(centers), p=weights)
        mu_x, mu_y = centers[center_idx]
        selected_sigma = sigma_list[center_idx]
        
        return self.gaussian_distri_generate(mu_x, mu_y, selected_sigma)


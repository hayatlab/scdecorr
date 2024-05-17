import torchvision.transforms as transforms
import numpy as np



class RandomSubArrayShuffle(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, x):
        n = len(x)
        idxs = np.array(range(n))
        k = np.int(np.floor(n * self.ratio))
        
        idxs1 = np.random.choice(idxs, k, replace=False)
        idxs2 = np.random.choice(np.setdiff1d(idxs, idxs1), k, replace=False)
        
        a = np.copy(x)
        a[idxs1] = x[idxs2]
        
        return a

class RandomZero(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, x):
        n = len(x)
        idxs = np.array(range(n))
        k = np.int(np.floor(n * self.ratio))
        
        idxs = np.random.choice(idxs, k, replace=False)
        
        a = np.copy(x)
        a[idxs] = 0
        
        return a

class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        return x + np.random.normal(loc=0.0, scale=self.std, size=len(x))

class Transform:
    def __init__(self):
        '''
        self.transform = transforms.Compose([
            RandomSubArrayShuffle(ratio=0.2),
            RandomGaussianNoise()
        ])
        self.transform_prime = transforms.Compose([
            RandomZero(),
            RandomGaussianNoise()
        ])
        '''
        self.transform = RandomSubArrayShuffle(ratio=0.2)
        self.transform_prime = RandomZero()

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return [y1, y2]
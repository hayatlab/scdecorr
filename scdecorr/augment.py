import torchvision.transforms as transforms
import numpy as np
 
 
class RandomShuffle(object):
    def __init__(self, ratio=0.1, p=0.5):
        self.ratio = ratio
        self.p = p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x

        n = len(x)
        k = int(np.floor(n * self.ratio))
        idx = np.random.choice(n, k, replace=False)

        a = np.copy(x)
        a[idx] = np.random.permutation(a[idx])
        return a


class RandomZero(object):
    def __init__(self, ratio=0.1, p=0.5):
        self.ratio = ratio
        self.p = p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x

        n = len(x)
        k = int(np.floor(n * self.ratio))
        idx = np.random.choice(n, k, replace=False)

        a = np.copy(x)
        a[idx] = 0
        return a


class RandomSwap(object):
    def __init__(self, ratio=0.1, p=0.5):
        self.ratio = ratio   # percentage of genes to swap (e.g., 0.1)
        self.p = p           # probability of applying transform

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x

        n = len(x)
        k = int(np.floor(n * self.ratio))
        # ensure even number for pairing
        if k % 2 != 0:
            k -= 1
        if k <= 0:
            return x

        idx = np.random.choice(n, k, replace=False)
        idx = idx.reshape(-1, 2)
        a = np.copy(x)
        # swap within each pair
        for i, j in idx:
            a[i], a[j] = a[j], a[i]
        return a


class RandomGaussianNoise(object):
    def __init__(self, ratio=0.1, mean=0.0, std=0.01, p=0.5):
        self.ratio = ratio
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x

        n = len(x)
        k = int(np.floor(n * self.ratio))
        idx = np.random.choice(n, k, replace=False)

        a = np.copy(x)
        noise = np.random.normal(self.mean, self.std, k)
        a[idx] += noise
        return a


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            RandomShuffle(ratio=0.1),
            RandomZero(ratio=0.2)
        ])
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return [y1, y2]




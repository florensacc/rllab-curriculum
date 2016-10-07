import matplotlib.pyplot as plt
import numpy as np

dir = '/Users/rein/Desktop/files/'
with open(dir + 'binary_codes.txt') as f:
    content = f.readlines()
    codes = np.vstack([np.array(list(str(c).rstrip('\n')), dtype=int) for c in content])
    plt.matshow(codes)
    plt.yticks(range(0, codes.shape[0], 5))
    plt.savefig(dir + 'mat.png', bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np

dir = '/Users/rein/Desktop/files/'
with open(dir + 'binary_code_0.txt') as f:
    content = f.readlines()
    codes = np.vstack([np.array(list(str(c).rstrip('\n')), dtype=int) for c in content])#[0:0+50]
    # plt.matshow(codes)
    plt.imshow(codes, cmap='Greys', vmin=-1, vmax=2, interpolation='nearest')
    plt.yticks(range(0, codes.shape[0], 10))
    # plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.gca().set_xticks([])
    plt.grid(which='minor')
    # plt.gca().tick_params(labelleft='off')
    # plt.gca().tick_params(axis='both', which='both', bottom='off', top='off',
    #                 labelbottom='off', right='off', left='off', labelleft='off')
    # plt.yticks(range(0, codes.shape[0], 10))
    # plt.yticks([])
    plt.savefig(dir + 'mat.png', bbox_inches='tight', dpi=300)

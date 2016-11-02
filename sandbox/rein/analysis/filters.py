import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(4, 8, top=1., bottom=0., right=1., left=0., hspace=0.,
                       wspace=0.1)

dir = '/Users/rein/Desktop/files/'
with open(dir + 'binary_code_0.txt') as f:
    content = f.readlines()
    codes = np.vstack([np.array(list(str(c).rstrip('\n')), dtype=int) for c in content])  # [0:0+50]
    codes = codes.reshape((-1, 5, 5, 32))
    for i, code in enumerate(codes):

        count = 0
        for g in gs:
            ax = plt.subplot(g)
            ax.imshow(code[:, :, count], cmap='Greys', vmin=-1, vmax=2, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_aspect('auto')
            # ax.axis('off')
            count += 1

        plt.savefig('/Users/rein/Desktop/tmp/filters{}.png'.format(i), dpi=300)

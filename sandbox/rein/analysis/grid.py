import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(5, 10, top=1., bottom=0., right=1., left=0., hspace=0.,
                       wspace=0.)

count = 0
for g in gs:
    img = mpimg.imread('/Users/rein/Desktop/tmp/actual/actual_{}_0.png'.format(count))
    ax = plt.subplot(g)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    ax.axis('off')
    count += 1

plt.savefig('/Users/rein/Desktop/tmp/fig.png', dpi=300)

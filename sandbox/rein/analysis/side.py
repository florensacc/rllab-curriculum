import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(1, 2, top=1., bottom=0., right=1., left=0., hspace=0.,
                       wspace=0.)

count = 0
for count in range(1000000):
    img = mpimg.imread('/Users/rein/Desktop/tmp/pred/model_{}_0.png'.format(count))
    ax = plt.subplot(gs[0])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('auto')
    ax.axis('off')
    img = mpimg.imread('/Users/rein/Desktop/tmp/filters{}.png'.format(count))
    ax = plt.subplot(gs[1])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('auto')
    ax.axis('off')

    plt.savefig('/Users/rein/Desktop/tmp/side{}.png'.format(count), dpi=300)
    count += 1
    plt.close()

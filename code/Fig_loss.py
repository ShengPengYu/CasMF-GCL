import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



if __name__ == "__main__":


    c=['m', 'b', 'c', 'g', 'k','r','y']
    fig,a = plt.subplots(2,1)
    fig.tight_layout(pad=2,h_pad=3,w_pad=3)
    data1_loss = np.loadtxt('./results/log/Disbiome_loss.txt', dtype=np.float32)
    data2_loss = np.loadtxt('./results/log/HMDAD_loss.txt', dtype=np.float32)

    x = range(10)

    a[0].plot(x, data1_loss, 'r-', label=u'CasMF-GCL')
    a[1].plot(x, data2_loss, 'g-', label=u'CasMF-GCL')

    a[0].legend()
    a[1].legend()
    # 显示图例
    # p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
    a[0].set_xlabel(u'iters')
    a[0].set_ylabel(u'loss')
    a[1].set_xlabel(u'iters')
    a[1].set_ylabel(u'loss')
    a[0].set_title('Loss on Disbiome')
    a[1].set_title('Loss on HMDAD')

    # axins.plot(x2,y2 , color='blue', ls='-')
    plt.rcParams['figure.figsize'] = (12.8, 7.2)
    plt.savefig("img_loss.pdf")
    plt.show()






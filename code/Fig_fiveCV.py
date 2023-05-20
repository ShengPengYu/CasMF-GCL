import matplotlib.pyplot as plt
import numpy as np

#aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision

def ParallelBar(ax, x_labels, y,ylim, labels=None, colors=None, width = 0.35, gap=2):
    '''
    绘制并排柱状图
    :param x_labels: list 横坐标刻度标识
    :param y: list 列表里每个小列表是一个系列的柱状图
    :param labels: list 每个柱状图的标签
    :param colors: list 每个柱状图颜色
    :param width: float 每条柱子的宽度
    :param gap: int 柱子与柱子间的宽度
    '''
    ax.set_ylim(ylim)
    # check params
    if labels is not None:
        if len(labels) < len(y): raise ValueError('labels的数目不足')
    if colors is not None:
        if len(colors) < len(y): raise ValueError('颜色colors的数目不足')
    if not isinstance(gap, int): raise ValueError('输入的gap必须为整数')

    x = [t for t in range(0, len(x_labels)*gap, gap)]  # the label locations
    for i in range(len(y)):
        if labels is not None: l = labels[i]
        else: l = None
        if colors is not None: color = colors[i]
        else: color = None
        if len(x) != len(y[i]): raise ValueError('所给数据数目与横坐标刻度数目不符')
        ax.bar(x, y[i], label=l, width=width, color=color)
        x = [t+width for t in x]
    x = [t + (len(y)-1)*width/2 for t in range(0, len(x_labels) * gap, gap)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(labels)


if __name__ == "__main__":
    x_labels = ['AUPR', 'AUC', 'F1', 'ACC', 'RECALL','SPEC','PREC']
    c=['m', 'b', 'c', 'g', 'k','r','y']
    y = np.loadtxt('./results/log/Disbiome_5kd.txt', dtype=np.float32)
    y1 = np.loadtxt('./results/log/HMDAD_5kd.txt', dtype=np.float32)
    labels = ['1-th', '2-th', '3-th', '4-th','5-th','MEAN']
    print(np.shape(y))

    fig,a = plt.subplots(2,1)
    fig.tight_layout(pad=2,h_pad=3,w_pad=3)
    #绘制平方函数
    print(np.shape(a))

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    ParallelBar(a[0],x_labels, y,[0.90,1], labels=labels, colors=c, width=0.35, gap=3)
    ParallelBar(a[1], x_labels, y1,[0.85,1], labels=labels, colors=c, width=0.35, gap=3)
    #绘制平方根图像
    #a[1].plot(x,np.sqrt(x))
    a[0].set_title('5-cv on Disbiome')
    a[1].set_title('5-cv on HMDAD')
    plt.savefig("img_5cv.pdf")
    plt.show()

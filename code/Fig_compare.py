import numpy as np
#绘制二分类ROC曲线
import matplotlib.pyplot as plt

from itertools import compress 



linewidth = '1' #//定义线条宽度
compare_labels = ["BiRWHMDA","GATMDA","RNMFMDA","BRWMDA","MVGCNMDA","MCHMDA","CasMF-GCL"]
compare_aucs = [0.880,0.932,0.907,0.908,0.919,0.923,0.997] #AUC值
compare_enable =[True,True,True,True,True,True,True,False,False,True,True]#是否参与对比试验
color=['crimson','seagreen','royalblue','peru','hotpink','tomato','black','magenta','gray','y','teal']





#绘制ROC曲线图
fig, ax = plt.subplots(1,1)
fig.tight_layout(pad=2,h_pad=3,w_pad=3)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.3, height*1.01, '%s' % float(height))


#绘制柱状图
index = range(len(compare_aucs))
width=0.35
bar_obj = plt.bar(list(compress(compare_labels,compare_enable)),list(compress(compare_aucs,compare_enable)),width,color=color)
plt.title('Performance comparison with other six methods on HMDAD dataset ')
# 设置横轴标签
plt.xlabel('Methods')
# 设置纵轴标签
plt.ylabel('AUC')
plt.ylim(0.75, 1.0)  

    
# 添加标题
#plt.set_title('Global AUC in difference methods')
plt.grid(False,linestyle='-.',linewidth=linewidth,color="grey")
autolabel(bar_obj)
# 添加图例
#ax[1].legend(loc="upper right")


fig.set_size_inches(11, 6)

plt.savefig("img_compare.pdf",bbox_inches='tight')
plt.show()
import numpy as np

import csv

if __name__ == "__main__":
    pre_matrix = np.loadtxt('./pre_matrix1.txt', dtype=np.float32)
    print(np.shape(pre_matrix))
    # Type 1 diabetes 35
    # Liver cirrhosis 25

    # Bacterial Vaginosis 7
    # Crohn's disease(CD) 13
    disease = pre_matrix[13,:]
    x = disease
    b = sorted(enumerate(x), key=lambda x:x[1])  # x[1]是因为在enumerate(a)中，a数值在第1位
    c = [x[0] for x in b]  # 获取排序好后b坐标,下标在第0位
    order = c[::-1]
    print(type(order))
    for i in range(len(order)):
        print(order[i])






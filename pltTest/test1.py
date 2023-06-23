import matplotlib.pyplot as plt

# 数据
# x = [1, 2, 3, 4, 5]
# y1 = [2, 4, 6, 8, 10]
# y2 = [1, 3, 5, 7, 9]
l1 = [0.2, 0.33, 0.444,0.346,0.412,0.564,0.666,0.789,0.888,0.99]
l2 = [0.3, 0.43, 0.444,0.546,0.552,0.564,0.62,0.71,0.87,0.96]

# 绘制折线图
# plt.plot(x, y1, color='blue', linestyle='dashed', label='Line 1')
# plt.plot(x, y2, color='red', linestyle='dotted', label='Line 2')

plt.plot(range(len(l1)), l1,label='Line 1')
plt.plot(range(len(l2)), l2,label='Line 2')

# 添加标题和图例
plt.title('Lines Plot')
plt.legend()

# # 设置横纵坐标范围
# plt.xlim(0, 6)
# plt.ylim(0, 12)

# 设置横纵坐标标签
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()
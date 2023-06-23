import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib
import pandas as pd
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
re1=pd.read_csv("./numUser_1_withoutFL.csv",index_col=None)
re2=pd.read_csv("./numUser_2_fedavg.csv",index_col=None)
re3=pd.read_csv("./numUser_3_fedavg.csv",index_col=None)
re4=pd.read_csv("./numUser_4_fedavg.csv",index_col=None)
re5=pd.read_csv("./numUser_5_fedavg.csv",index_col=None)
re6=pd.read_csv("./numUser_6_fedavg.csv",index_col=None)
pre2=pd.read_csv("./numUser_2_fedProx.csv",index_col=None)
pre3=pd.read_csv("./numUser_3_fedProx.csv",index_col=None)
pre4=pd.read_csv("./numUser_4_fedProx.csv",index_col=None)
pre5=pd.read_csv("./numUser_5_fedProx.csv",index_col=None)
pre6=pd.read_csv("./numUser_6_fedProx.csv",index_col=None)
sns.set_style("ticks")
# _,(ax0, ax1,ax2) = plt.subplots(1,3,figsize=(15,5))
# _,(ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14) = plt.subplots(5,3,figsize=(15,5))
# _,axs = plt.subplots(5,3,figsize=(15,5))
p1 = plt.figure(1)
axs = p1.subplots(1,3)
# ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14 = axs.flatten()
ax0, ax1,ax2 = axs.flatten()
ax0.grid()
ax1.grid()
ax2.grid()
# ax3.grid()
# ax4.grid()
# ax5.grid()
# ax6.grid()
# ax7.grid()
# ax8.grid()
# ax9.grid()
# ax10.grid()
# ax11.grid()
# ax12.grid()
# ax13.grid()
# ax14.grid()
# ax2[1].grid()
# plt.figure(1)
ax0.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax0.plot(re2['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax0.plot(pre2['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax0.set_xlabel('epoch')  # Add an x-label to the axes.
ax0.set_ylabel('loss')  # Add a y-label to the axes.
ax0.set_title("Loss(Client2)")
ax0.legend()

ax1.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
ax1.plot(re2['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax1.plot(pre2['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax1.set_xlabel('epoch')  # Add an x-label to the axes.
ax1.set_ylabel('train_acc')  # Add a y-label to the axes.
ax1.set_title("Train Accuracy(Client2)")
ax1.legend()

ax2.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
ax2.plot(re2['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax2.plot(pre2['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax2.set_xlabel('epoch')  # Add an x-label to the axes.
ax2.set_ylabel('test_acc')  # Add a y-label to the axes.
ax2.set_title("Test Accuracy(Client2)")
ax2.legend()

p1.suptitle("Different Fed Algorithms",fontsize='x-large')
p1.savefig('./FedDiffPlotClient2.png')

# plt.figure(2)
p2 = plt.figure(2)
# _,axs = p2.subplots(1,3)
axs = p2.subplots(1,3)
# ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14 = axs.flatten()
ax0, ax1,ax2 = axs.flatten()
ax0.grid()
ax1.grid()
ax2.grid()
ax0.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax0.plot(re3['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax0.plot(pre3['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax0.set_xlabel('epoch')  # Add an x-label to the axes.
ax0.set_ylabel('loss')  # Add a y-label to the axes.
ax0.set_title("Loss(Client3)")
ax0.legend()

ax1.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
ax1.plot(re3['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax1.plot(pre3['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax1.set_xlabel('epoch')  # Add an x-label to the axes.
ax1.set_ylabel('train_acc')  # Add a y-label to the axes.
ax1.set_title("Train Accuracy(Client3)")
ax1.legend()

ax2.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
ax2.plot(re3['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax2.plot(pre3['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax2.set_xlabel('epoch')  # Add an x-label to the axes.
ax2.set_ylabel('test_acc')  # Add a y-label to the axes.
ax2.set_title("Test Accuracy(Client3)")
ax2.legend()

p2.suptitle("Different Fed Algorithms",fontsize='x-large')
p2.savefig('./FedDiffPlotClient3.png')

# plt.figure(3)
p3 = plt.figure(3)
axs = p3.subplots(1,3)
# ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14 = axs.flatten()
ax0, ax1,ax2 = axs.flatten()
ax0.grid()
ax1.grid()
ax2.grid()
ax0.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax0.plot(re4['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax0.plot(pre4['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax0.set_xlabel('epoch')  # Add an x-label to the axes.
ax0.set_ylabel('loss')  # Add a y-label to the axes.
ax0.set_title("Loss(Client4)")
ax0.legend()

ax1.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
ax1.plot(re4['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax1.plot(pre4['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax1.set_xlabel('epoch')  # Add an x-label to the axes.
ax1.set_ylabel('train_acc')  # Add a y-label to the axes.
ax1.set_title("Train Accuracy(Client4)")
ax1.legend()

ax2.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
ax2.plot(re4['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax2.plot(pre4['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax2.set_xlabel('epoch')  # Add an x-label to the axes.
ax2.set_ylabel('test_acc')  # Add a y-label to the axes.
ax2.set_title("Test Accuracy(Client4)")
ax2.legend()

p3.suptitle("Different Fed Algorithms",fontsize='x-large')
p3.savefig('./FedDiffPlotClient4.png')


# plt.figure(4)
p4 = plt.figure(4)
axs = p4.subplots(1,3)
# ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14 = axs.flatten()
ax0, ax1,ax2 = axs.flatten()
ax0.grid()
ax1.grid()
ax2.grid()
ax0.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax0.plot(re5['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax0.plot(pre5['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax0.set_xlabel('epoch')  # Add an x-label to the axes.
ax0.set_ylabel('loss')  # Add a y-label to the axes.
ax0.set_title("Loss(Client5)")
ax0.legend()

ax1.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
ax1.plot(re5['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax1.plot(pre5['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax1.set_xlabel('epoch')  # Add an x-label to the axes.
ax1.set_ylabel('train_acc')  # Add a y-label to the axes.
ax1.set_title("Train Accuracy(Client5)")
ax1.legend()

ax2.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
ax2.plot(re5['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax2.plot(pre5['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax2.set_xlabel('epoch')  # Add an x-label to the axes.
ax2.set_ylabel('test_acc')  # Add a y-label to the axes.
ax2.set_title("Test Accuracy(Client5)")
ax2.legend()

p4.suptitle("Different Fed Algorithms",fontsize='x-large')
p4.savefig('./FedDiffPlotClient5.png')


# plt.figure(5)
p5 = plt.figure(5)
axs = p5.subplots(1,3)
# ax0, ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14 = axs.flatten()
ax0, ax1,ax2 = axs.flatten()
ax0.grid()
ax1.grid()
ax2.grid()
ax0.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax0.plot(re6['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax0.plot(pre6['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax0.set_xlabel('epoch')  # Add an x-label to the axes.
ax0.set_ylabel('loss')  # Add a y-label to the axes.
ax0.set_title("Loss(Client6)")
ax0.legend()

ax1.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
ax1.plot(re6['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax1.plot(pre6['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax1.set_xlabel('epoch')  # Add an x-label to the axes.
ax1.set_ylabel('train_acc')  # Add a y-label to the axes.
ax1.set_title("Train Accuracy(Client6)")
ax1.legend()

ax2.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
ax2.plot(re6['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax2.plot(pre6['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")

ax2.set_xlabel('epoch')  # Add an x-label to the axes.
ax2.set_ylabel('test_acc')  # Add a y-label to the axes.
ax2.set_title("Test Accuracy(Client6)")
ax2.legend()

p5.suptitle("Different Fed Algorithms",fontsize='x-large')
p5.savefig('./FedDiffPlotClient6.png')


# plt.suptitle("Different Fed Algorithms",fontsize='x-large')
# plt.savefig('./FedDiffClient2-6Plot.png')
# plt.show()
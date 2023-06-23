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
# _,(ax0, ax1,ax2,ax3) = plt.subplots(2,2,figsize=(15,5))
_,axs = plt.subplots(2,2,figsize=(15,5))
ax0, ax1,ax2,ax3 = axs.flatten()
# _,(ax0, ax1,ax2),(ax3,ax4,ax5),(ax6,ax7,ax8),(ax9,ax10,ax11),(ax12,ax13,ax14) = plt.subplots(5,3,figsize=(15,5))
ax0.grid()
ax1.grid()
ax2.grid()
ax3.grid()
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


ax3.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
ax3.plot(re3['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
ax3.plot(pre3['loss'],linestyle='-',marker='.',label="FedProx",color="green")

ax3.set_xlabel('epoch')  # Add an x-label to the axes.
ax3.set_ylabel('loss')  # Add a y-label to the axes.
ax3.set_title("Loss(Client3)")
ax3.legend()

# ax4.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax4.plot(re3['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax4.plot(pre3['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax4.set_xlabel('epoch')  # Add an x-label to the axes.
# ax4.set_ylabel('train_acc')  # Add a y-label to the axes.
# ax4.set_title("Train Accuracy(Client3)")
# ax4.legend()
#
# ax5.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax5.plot(re3['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax5.plot(pre3['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax5.set_xlabel('epoch')  # Add an x-label to the axes.
# ax5.set_ylabel('test_acc')  # Add a y-label to the axes.
# ax5.set_title("Test Accuracy(Client3)")
# ax5.legend()
#
# ax6.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
# ax6.plot(re4['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax6.plot(pre4['loss'],linestyle='-',marker='.',label="FedProx",color="green")
#
# ax6.set_xlabel('epoch')  # Add an x-label to the axes.
# ax6.set_ylabel('loss')  # Add a y-label to the axes.
# ax6.set_title("Loss(Client4)")
# ax6.legend()
#
# ax7.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax7.plot(re4['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax7.plot(pre4['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax7.set_xlabel('epoch')  # Add an x-label to the axes.
# ax7.set_ylabel('train_acc')  # Add a y-label to the axes.
# ax7.set_title("Train Accuracy(Client4)")
# ax7.legend()
#
# ax8.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax8.plot(re4['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax8.plot(pre4['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax8.set_xlabel('epoch')  # Add an x-label to the axes.
# ax8.set_ylabel('test_acc')  # Add a y-label to the axes.
# ax8.set_title("Test Accuracy(Client4)")
# ax8.legend()
#
#
# ax9.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
# ax9.plot(re5['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax9.plot(pre5['loss'],linestyle='-',marker='.',label="FedProx",color="green")
#
# ax9.set_xlabel('epoch')  # Add an x-label to the axes.
# ax9.set_ylabel('loss')  # Add a y-label to the axes.
# ax9.set_title("Loss(Client5)")
# ax9.legend()
#
# ax10.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax10.plot(re5['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax10.plot(pre5['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax10.set_xlabel('epoch')  # Add an x-label to the axes.
# ax10.set_ylabel('train_acc')  # Add a y-label to the axes.
# ax10.set_title("Train Accuracy(Client5)")
# ax10.legend()
#
# ax11.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax11.plot(re5['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax11.plot(pre5['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax11.set_xlabel('epoch')  # Add an x-label to the axes.
# ax11.set_ylabel('test_acc')  # Add a y-label to the axes.
# ax11.set_title("Test Accuracy(Client5)")
# ax11.legend()
#
#
# ax12.plot(re1['loss'],linestyle='-',marker='.',label="without FL",color="red")
# ax12.plot(re6['loss'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax12.plot(pre6['loss'],linestyle='-',marker='.',label="FedProx",color="green")
#
# ax12.set_xlabel('epoch')  # Add an x-label to the axes.
# ax12.set_ylabel('loss')  # Add a y-label to the axes.
# ax12.set_title("Loss(Client6)")
# ax12.legend()
#
# ax13.plot(re1['trainacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax13.plot(re6['trainacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax13.plot(pre6['trainacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax13.set_xlabel('epoch')  # Add an x-label to the axes.
# ax13.set_ylabel('train_acc')  # Add a y-label to the axes.
# ax13.set_title("Train Accuracy(Client6)")
# ax13.legend()
#
# ax14.plot(re1['testacc'],linestyle='-',marker='.',label="without FL",color="red")
# ax14.plot(re6['testacc'],linestyle='-',marker='.',label="FedAvg",color="blue")
# ax14.plot(pre6['testacc'],linestyle='-',marker='.',label="FedAvg",color="green")
#
# ax14.set_xlabel('epoch')  # Add an x-label to the axes.
# ax14.set_ylabel('test_acc')  # Add a y-label to the axes.
# ax14.set_title("Test Accuracy(Client6)")
# ax14.legend()



plt.suptitle("Different Fed Algorithms",fontsize='x-large')
plt.savefig('./testPlot.png')
plt.show()
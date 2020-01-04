import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 解决matplotlib中文问题
from pylab import mpl
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
# 读入数据集
df_u = pd.read_csv('data/heart.csv')

# 显示数据集的具体信息
def DataInfo(df):
    buf = io.StringIO()
    df.info(buf=buf)
    info = buf.getvalue()
    return info,df.describe()

# 显示数据集中未得病和的病的比例表格
def SickDataGraph():
    sns.countplot(x='target', data=df_u, palette="muted")
    plt.xlabel("未得病/得病比例")
    plt.savefig("sick.png")


# 显示数据集中男女比例
def SexDataGraph():
    sns.countplot(x='sex', data=df_u, palette="Set3")
    plt.xlabel("Sex (0 = 女, 1= 男)")
    plt.savefig("Sex.png")


# 显示所有数据情况
def DataTotalGraph():
    plt.figure(figsize=(18, 7))
    sns.countplot(x='age', data=df_u, hue='target', palette='PuBuGn', saturation=0.8)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig("total.png")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import data_info
# 解决matplotlib中文问题
from pylab import mpl
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

def DataCut():
    # 读入数据集
    df = data_info.df_u
    first = pd.get_dummies(df['cp'], prefix="cp")
    second = pd.get_dummies(df['slope'], prefix="slope")
    third = pd.get_dummies(df['thal'], prefix="thal")
    df = pd.concat([df, first, second, third], axis=1)
    df = df.drop(columns=['cp', 'slope', 'thal'])
    y = df.target.values
    X = df.drop(['target'], axis=1)
    # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test)
    return X_train,X_test,y_train,y_test,X,y,df

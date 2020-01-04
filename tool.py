import matplotlib.pyplot as plt
import seaborn as sns
import data_treatment
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from pylab import mpl
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

# 获取训练测试集
X_train,X_test,y_train,y_test,X,y,df = data_treatment.DataCut()
# Log = LogisticRegression.Logistic()

# DecisonT = DecisionTree.DeTree()
# 训练模型
def TrainModel(model):
    model.fit(X_train, y_train)
# 交叉验证概率
def Cross_val_score(model):
    return cross_val_score(model, X, y, cv=5).mean()

# 训练集概率
def Train_score(model):
    return model.score(X_train,y_train)
# 测试集概率
def Test_score(model):
    return model.score(X_test,y_test)

# 使用网格搜索找出更好的模型参数
def BetterModel(param_grid,model):
    grid_search = GridSearchCV(model, param_grid, cv=10, n_jobs=-1)
    TrainModel(grid_search)
    bestModel = grid_search.best_estimator_
    bestScore = grid_search.best_score_
    return bestModel,bestScore
# 调用accuracy_score计算分类准确度
def Count_accuracy_score(model):
    y_predict_log = model.predict(X_test)
    return accuracy_score(y_test,y_predict_log)

# 显示评价
def ShowPreRecallF1sc(model):
    y_predict = model.predict(X_test)
    return classification_report(y_test,y_predict)

# 显示LGRoc曲线

def showAllGraph(LogisticRegression,Knn,DecisionTree):
    decision_scores = LogisticRegression.log_reg.decision_function(X_test)
    fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
    y_probabilities = Knn.knn_clf.predict_proba(X_test)[:, 1]
    fprs1, tprs1, thresholds1 = roc_curve(y_test, y_probabilities)
    y_probabilities = DecisionTree.dt_clf.predict_proba(X_test)[:, 1]
    fprs2, tprs2, thresholds2 = roc_curve(y_test, y_probabilities)
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    plt.title('ROC Curve', fontsize=18)
    plt.plot(fprs1, tprs1, label='KNN')
    plt.plot(fprs, tprs, label='Log_Reg')
    plt.plot(fprs2, tprs2, label='dt_Clf')
    plt.plot([0, 1], ls='--')
    plt.plot([0, 0], [1, 0], c='.8')
    plt.plot([1, 1], c='.8')
    plt.ylabel('TP rate', fontsize=15)
    plt.xlabel('FP rate', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig("Roc.png")
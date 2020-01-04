from sklearn.linear_model import LogisticRegression
import tool
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
class Logistic:
 # 训练模型
    log_reg = LogisticRegression(solver='liblinear')
    def __init__(self):
        self.param_grid = [
             {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2', 'l1'],
                'class_weight': ['balanced', None]
            }
            ]
    def set(self,model):
        self.log_reg = model

    def trainLog(self):
        tool.TrainModel(self.log_reg)


    def getScore(self):
        cross_score = tool.Cross_val_score(self.log_reg)
        Train_score = tool.Train_score(self.log_reg)
        Test_score = tool.Test_score(self.log_reg)
        Count_accuracy_score = tool.Count_accuracy_score(self.log_reg)
        return cross_score,Train_score,Test_score,Count_accuracy_score


    def showTarget(self):
        Target = tool.ShowPreRecallF1sc(self.log_reg)
        return Target


    # 使用网格搜索找出更好的模型参数

    def Better_ModelLg(self):
        bestModel, bestScore = tool.BetterModel(self.param_grid, self.log_reg)
        self.set(bestModel)
        return bestModel,bestScore


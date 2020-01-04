from sklearn import neighbors
import tool
class Knn:
    knn_clf = neighbors.KNeighborsClassifier()
    def __init__(self):
        self.param_grid = [
            {
                'weights': ['uniform'],
                'n_neighbors': [i for i in range(1, 31)]
            },
            {
                'weights': ['distance'],
                'n_neighbors': [i for i in range(1, 31)],
                'p': [i for i in range(1, 6)]
            }
        ]
    def set(self,model):
        self.knn_clf = model

    def train_knn(self):
        tool.TrainModel(self.knn_clf)
    # 分数测试
    def getScore(self):
        cross_score = tool.Cross_val_score(self.knn_clf)
        Train_score = tool.Train_score(self.knn_clf)
        Test_score = tool.Test_score(self.knn_clf)
        Count_accuracy_score = tool.Count_accuracy_score(self.knn_clf)
        return cross_score,Train_score,Test_score,Count_accuracy_score
    def showTarget(self):
        Target = tool.ShowPreRecallF1sc(self.knn_clf)
        return Target

    def Better_ModelKNN(self):
        # global knn_clf
        bestModel, bestScore = tool.BetterModel(self.param_grid, self.knn_clf)
        self.set(bestModel)
        #knn_clf = bestModel
        return bestModel,bestScore
from sklearn.tree import DecisionTreeClassifier
import tool

class DeTree:
    dt_clf = DecisionTreeClassifier()
    def __init__(self):
        self.param_grid = [
            {
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }
        ]
    def set(self,model):
        self.dt_clf = model

    def train_decisionTree(self):
        tool.TrainModel(self.dt_clf)

    def getScore(self):
        cross_score = tool.Cross_val_score(self.dt_clf)
        Train_score = tool.Train_score(self.dt_clf)
        Test_score = tool.Test_score(self.dt_clf)
        Count_accuracy_score = tool.Count_accuracy_score(self.dt_clf)
        return cross_score,Train_score,Test_score,Count_accuracy_score

    def showTarget(self):
        Target = tool.ShowPreRecallF1sc(self.dt_clf)
        return Target

    def Better_ModelDST(self):
        bestModel, bestScore = tool.BetterModel(self.param_grid, self.dt_clf)
        self.set(bestModel)
        return bestModel,bestScore
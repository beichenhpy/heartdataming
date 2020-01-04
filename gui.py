import tkinter as tk
import tool
import data_treatment
import data_info
import KNN
import LogisticRegression
import DecisionTree
from PIL import Image
from tkinter import scrolledtext
from tkinter import ttk
class GUI(tk.Frame):
    def __init__(self,win):
        tk.Frame.__init__(self,win)
        frame_top = tk.Frame(self)
        frame_bottom = tk.Frame(self)
        win.title("数据挖掘-分类算法 v1.0---------------------------------------------------------------------作者：韩鹏宇")
        win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="10", pady="10")
        frame_top.pack(side='top', expand=1, fill=tk.Y)
        frame_bottom.pack(side='bottom', expand=1, fill=tk.X)
        self.show_label =scrolledtext.ScrolledText(frame_top,width=100, height=20)
        self.show_label.pack(side = 'top')
        self.show_info = tk.Button(frame_bottom,text = '显示数据集信息',width=15,height=3,command=self.show_info)
        self.show_info.pack(side=tk.LEFT)
        self.show_dataGraph = tk.Button(frame_bottom,text = '显示数据集关系图',width=15,height=3,command=self.show_dataGrapgh)
        self.show_dataGraph.pack(side = tk.LEFT)
        number = tk.StringVar()
        self.model = ttk.Combobox(frame_bottom,width=12, textvariable=number)
        self.model.pack(side = tk.LEFT)
        self.model['values'] = ['逻辑回归','KNN','决策树']
        self.train = tk.Button(frame_bottom,text ='训练',width=15,height=3,command = self.getModelName)
        self.train.pack(side = tk.LEFT)
        self.score = tk.Button(frame_bottom,text = '准确率',width=15,height=3,command = self.getScore)
        self.score.pack(side = tk.LEFT)
        self.showTarget = tk.Button(frame_bottom,text = '查看指标',width=15,height=3,command = self.showTarget)
        self.showTarget.pack(side = tk.LEFT)
        self.BetterMode = tk.Button(frame_bottom,text = '获得更好的参数模型',width = 20,height = 3 ,command = self.BetterMode)
        self.BetterMode.pack(side = tk.LEFT)
        self.showRog = tk.Button(frame_bottom,text = '显示总Roc图像',width = 15,height = 3,command = self.showRog)
        self.showRog.pack(side = tk.LEFT)
        self.modelName = ''
        self.flag2 = True
        self.Log = LogisticRegression.Logistic()
        self.Knn = KNN.Knn()
        self.DecTrees = DecisionTree.DeTree()
    def show_info(self):
        if self.flag2 is True:
            info = data_info.DataInfo(data_info.df_u)
            self.show_label.insert(tk.END,info)
        else:
            X_train, X_test, y_train, y_test, X, y, df = data_treatment.DataCut()
            info = data_info.DataInfo(df)
            self.show_label.insert(tk.END, info)
            self.flag2 = True
    def show_dataGrapgh(self):
        data_info.SickDataGraph()
        data_info.SexDataGraph()
        data_info.DataTotalGraph()
        images = ['sick.png','Sex.png','total.png']
        for i in images:
            im = Image.open(i)
            im.show()
    def getModelName(self):
        self.flag2 = False
        self.modelName = self.model.get()
        if self.modelName == '逻辑回归':
            self.Log.trainLog()
        elif self.modelName == 'KNN':
            self.Knn.train_knn()
        else:
            self.DecTrees.train_decisionTree()
    def tool1(self,cross_score,Train_score,Test_score,Count_accuracy_score):
        self.show_label.insert(tk.END, '\n交叉验证概率：')
        self.show_label.insert(tk.END, cross_score)
        self.show_label.insert(tk.END, '\n训练集验证概率：')
        self.show_label.insert(tk.END, Train_score)
        self.show_label.insert(tk.END, '\n测试集验证概率：')
        self.show_label.insert(tk.END, Test_score)
        self.show_label.insert(tk.END, '\naccuracy_score分类准确概率：')
        self.show_label.insert(tk.END, Count_accuracy_score)
    def tool2(self, bestModel,bestScore):
        self.show_label.insert(tk.END,'\n更好的模型参数：')
        self.show_label.insert(tk.END,bestModel)
        self.show_label.insert(tk.END, '\n更好的模型准确率：')
        self.show_label.insert(tk.END, bestScore)
    def getScore(self):
        self.modelName = self.model.get()
        if self.modelName == '逻辑回归':
            cross_score,Train_score,Test_score,Count_accuracy_score=self.Log.getScore()
            self.tool1(cross_score,Train_score,Test_score,Count_accuracy_score)
        elif self.modelName == 'KNN':
            cross_score, Train_score, Test_score, Count_accuracy_score = self.Knn.getScore()
            self.tool1(cross_score, Train_score, Test_score, Count_accuracy_score)
        else:
            cross_score, Train_score, Test_score, Count_accuracy_score = self.DecTrees.getScore()
            self.tool1(cross_score, Train_score, Test_score, Count_accuracy_score)
    def showTarget(self):
        self.modelName = self.model.get()
        if self.modelName == '逻辑回归':
            Target = self.Log.showTarget()
            self.show_label.insert(tk.END, '\n指标：')
            self.show_label.insert(tk.END, Target)
        elif self.modelName == 'KNN':
            Target = self.Knn.showTarget()
            self.show_label.insert(tk.END, '\n指标：')
            self.show_label.insert(tk.END, Target)
        else:
            Target = self.DecTrees.showTarget()
            self.show_label.insert(tk.END, '\n指标：')
            self.show_label.insert(tk.END, Target)
    def BetterMode(self):
        self.modelName = self.model.get()
        if self.modelName == '逻辑回归':
            bestModel,bestScore = self.Log.Better_ModelLg()
            self.tool2(bestModel,bestScore)
        elif self.modelName == 'KNN':
            bestModel, bestScore = self.Knn.Better_ModelKNN()
            self.tool2(bestModel, bestScore)
        else:
            bestModel, bestScore = self.DecTrees.Better_ModelDST()
            self.tool2(bestModel, bestScore)
    def showRog(self):
        tool.showAllGraph(self.Log,self.Knn,self.DecTrees)
        im = Image.open('Roc.png')
        im.show()
if __name__ == '__main__':
    win = tk.Tk()
    gui = GUI(win)
    win.mainloop()
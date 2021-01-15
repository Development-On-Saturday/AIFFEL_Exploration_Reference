from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from pandas import DataFrame
import pandas as pd

def call_big5_model():
    decision_model = DecisionTreeClassifier()
    randomforest_model = RandomForestClassifier()
    svm_model = svm.SVC()
    sgd_model = SGDClassifier()
    logistic_model = LogisticRegression()
    
    model = [decision_model, randomforest_model, svm_model, sgd_model, logistic_model]
    
    return model

def ml_model(data, model):
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    
    y_preds = {}
    accuracies = {}
    scores = []
    n = len(model)
    
    for m in model:
        m.fit(x_train, y_train)
        y_preds["y_pred_"+str(m)] = m.predict(x_test)
        accuracies["accu_"+str(m)] = accuracy_score(y_test, y_preds["y_pred_"+str(m)])
        scores.append(accuracy_score(y_test, y_preds["y_pred_"+str(m)]))
    
    
    print(str(n) + "가지 model 학습완료")
    
    accuracies = {'accuracy':scores}
    accuracies_frame = DataFrame(accuracies, columns = ['accuracy'],
                             index = ['decision','randomforest','svm','sgd','logistic'])
    
    return y_preds, accuracies, accuracies_frame


# library
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# sklearn classifier
class SklearnModels:
    '''
    필요한 모델을 선택하는 클래스
    '''
    def model_select(self, model_name, params={"random_state" : 123}):
        '''
        model_name:
            Decision Tree : DT
            Random Forest : RF
            SVM : SVM
            SGD : SGD
            Logistic Regression : LG
        '''
        if model_name == 'DT':
            model = DecisionTreeClassifier(**params)
        elif model_name == "RF":
            model = RandomForestClassifier(**params)
        elif model_name == "SVM":
            model = SVC(**params)
        elif model_name == 'SGD':
            model = SGDClassifier(**params)
        elif model_name =='LG':
            model = LogisticRegression(**params)

        return model

# 모델을 학습하고 테스트하는 클래스
class TrainTestSklClassifier(SklearnModels):
    '''
    데이터를 받아서 모델을 학습하고 테스트할 수 있는 클래스
    '''
    
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=123 ,test_size=0.3, stratify=y, # label 별로 골고루 균등하게 나누기 위해 설정
        )
        # 3차원 이상의 데이터 
        if self.X_train.ndim >=3:
            n,m = self.X_train.shape[1:]
            self.X_train = self.X_train.reshape((-1, n*m))
            self.X_test = self.X_test.reshape((-1, n*m))

    def train(self, model_name, params={"random_state" : 123}):
        '''
        모델 학습
        '''
        self.model = self.model_select(model_name, params)
        self.model.fit(self.X_train, self.y_train)
        return self.model 

    def test(self, metric="cr"):
        ''' 
        모델 테스트
        '''
        y_pred = self.model.predict(self.X_test)
        
        # metric 선택
        if metric == 'acc':
              result = accuracy_score(self.y_test , y_pred)
        
        elif metric == 'cr':
            result = classification_report(self.y_test, y_pred)
        
        elif metric =='f1':
            if pd.Series(self.y_test).nunique() >=3:
                result = f1_score(self.y_test, y_pred, average='macro')
            else:
                result = f1_score(self.y_test, y_pred)
                
        if type(result) != str:
            print('{} 결과 : {:.3f}'.format(metric, result))
        else:
            print('{} 결과 : {}'.format(metric, result))

        print()
        
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_classif

class Result:
    def __init__(self,datapath):
        self.datapath = datapath

    def read_data(self):
        df = pd.read_csv(self.datapath,header=None)
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        return (x,y)
    
    def split(self,x,y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return(X_train, X_test, y_train, y_test)
    
    def standardization(self,X_train,X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(X_train_scaled)
        print(X_test_scaled)
        return (X_train_scaled,X_test_scaled)
    
class Regularizer:
    def __init__(self,X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def lasso(self):
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(self.X_train_scaled, self.y_train)
        coefficients = lasso_model.coef_
        
        print(coefficients)
        X_train_lasso = self.X_train_scaled * coefficients
        X_train_lasso = pd.DataFrame(X_train_lasso)
        print(X_train_lasso)
        zero_columns_train = X_train_lasso.columns[(X_train_lasso == 0).all()]
        X_train_lasso.drop(zero_columns_train, axis=1, inplace=True)
        X_test_lasso = self.X_test_scaled * coefficients
        #X_train_scaled can be initialize to X_train_lasso to remove those columns eliminated by Lasso
        X_test_lasso = pd.DataFrame(X_test_lasso)
        zero_columns_test = X_test_lasso.columns[(X_test_lasso == 0).all()]
        X_test_lasso.drop(zero_columns_test, axis=1, inplace=True)

        return (X_train_lasso, X_test_lasso)
    
    def ridge(self):
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(self.X_train_scaled, self.y_train)
        coefficients = ridge_model.coef_
        
        X_train_ridge = self.X_train_scaled * coefficients
        X_test_ridge = self.X_test_scaled * coefficients

        return (X_train_ridge, X_test_ridge)

    def pls(self):
        pls_model = PLSRegression(n_components=10)
        pls_model.fit(self.X_train_scaled, self.y_train)
        weights = pls_model.coef_

        return weights
    
    def mutual_information(self):
        mi_scores = mutual_info_classif(self.X_train_scaled, self.y_train)
        return mi_scores

    
class saver:
    def __init__(self,datapath,data,regularizer):
        self.datapath = datapath
        self.data = data
        self.regularizer = regularizer
        self.dataframe = pd.read_excel(self.datapath)
    def save(self):
        frame = self.dataframe
        data = {'Regularizer': self.regularizer, 'Accuracy': self.data['Accuracy'], 'Error': 1-self.data['Accuracy'], 'Precision':self.data['Precision'], 'Recall':self.data['Recall'], 'F-measure':self.data['f-score']}
        frame = frame._append(data, ignore_index=True)
        frame.to_excel(self.datapath, index=False)
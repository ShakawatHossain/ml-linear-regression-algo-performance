from result import Result, Regularizer, saver
from classifier import Classifier
import numpy as np

new_obj = Result('ML_Regression/Dataset/ReLink/Apache.csv')
x,y = new_obj.read_data()
X_train, X_test, y_train, y_test = new_obj.split(x,y)
X_train_scaled,X_test_scaled = new_obj.standardization(X_train,X_test)

# Lasso
lasso_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
train_feature_lasso,test_feature_lasso = lasso_model.lasso()

clasification = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)
svm = clasification.svm()
knn = clasification.knn()
lda = clasification.lda()

lasso_svm = saver('ML_Regression/svm.xlsx',svm,'Lasso')
lasso_svm.save()

lasso_knn = saver('ML_Regression/knn.xlsx',knn,'Lasso')
lasso_knn.save()

lasso_lda = saver('ML_Regression/lda.xlsx',lda,'Lasso')
lasso_lda.save()



#Ridge

ridge_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
train_feature_ridge,test_feature_ridge = ridge_model.ridge()

clasification = Classifier(train_feature_ridge,test_feature_ridge,y_train, y_test)
svm = clasification.svm()
knn = clasification.knn()
lda = clasification.lda()

ridge_svm = saver('ML_Regression/svm.xlsx',svm,'Ridge')
ridge_svm.save()

ridge_knn = saver('ML_Regression/knn.xlsx',knn,'Ridge')
ridge_knn.save()

ridge_lda = saver('ML_Regression/lda.xlsx',lda,'Ridge')
ridge_lda.save()


#PLS

pls_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
feature_weights = pls_model.pls()
weights = []
for index,x in enumerate(feature_weights):
        weights.insert(index,x[0])

sorted_weights = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)
selected_features = sorted_weights[:10]
feature_index = []
for index,y in enumerate(selected_features):
    feature_index.insert(index,y[0])

train_feature_pls = X_train[X_train.columns[feature_index]]
test_feature_pls = X_test[X_test.columns[feature_index]]

clasification = Classifier(train_feature_pls,test_feature_pls,y_train,y_test)
svm = clasification.svm()
knn = clasification.knn()
lda = clasification.lda()

pls_svm = saver('ML_Regression/svm.xlsx',svm,'PLS')
pls_svm.save()

pls_knn = saver('ML_Regression/knn.xlsx',knn,'PLS')
pls_knn.save()

pls_lda = saver('ML_Regression/lda.xlsx',lda,'PLS')
pls_lda.save()


#Mutual Information
mi_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
mi_scores = pls_model.mutual_information()

scores = []
for index,x in enumerate(mi_scores):
        scores.insert(index,x)

sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
selected_features = sorted_scores[:10]
feature_index = []
for index,y in enumerate(selected_features):
    feature_index.insert(index,y[0])

train_feature_mi = X_train[X_train.columns[feature_index]]
test_feature_mi = X_test[X_test.columns[feature_index]]

clasification = Classifier(train_feature_mi,test_feature_mi,y_train,y_test)
svm = clasification.svm()
knn = clasification.knn()
lda = clasification.lda()


mi_svm = saver('ML_Regression/svm.xlsx',svm,'MI')
mi_svm.save()

mi_knn = saver('ML_Regression/knn.xlsx',knn,'MI')
mi_knn.save()

mi_lda = saver('ML_Regression/lda.xlsx',lda,'MI')
mi_lda.save()
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

class Classifier:
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def svm(self):
        svm_classifier = SVC()
        svm_classifier.fit(self.x_train, self.y_train)
        predictions_svm = svm_classifier.predict(self.x_test)

        precision = precision_score(self.y_test, predictions_svm)
        recall = recall_score(self.y_test, predictions_svm)
        accuracy = accuracy_score(self.y_test, predictions_svm)
        f1_measure = f1_score(self.y_test, predictions_svm)

        result_dict = {'Accuracy':accuracy, 'Precision': precision, 'Recall':recall, 'f-score': f1_measure}
        return result_dict
    
    def knn(self):
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(self.x_train, self.y_train)
        predictions_knn = knn_classifier.predict(self.x_test)

        precision = precision_score(self.y_test, predictions_knn)
        recall = recall_score(self.y_test, predictions_knn)
        accuracy = accuracy_score(self.y_test, predictions_knn)
        f1_measure = f1_score(self.y_test, predictions_knn)

        result_dict = {'Accuracy':accuracy, 'Precision': precision, 'Recall':recall, 'f-score': f1_measure}
        return result_dict
    
    def lda(self):
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(self.x_train, self.y_train)
        predictions_lda = lda_model.predict(self.x_test)

        precision = precision_score(self.y_test, predictions_lda)
        recall = recall_score(self.y_test, predictions_lda)
        accuracy = accuracy_score(self.y_test, predictions_lda)
        f1_measure = f1_score(self.y_test, predictions_lda)

        result_dict = {'Accuracy':accuracy, 'Precision': precision, 'Recall':recall, 'f-score': f1_measure}
        return result_dict

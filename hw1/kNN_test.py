from sklearn.metrics import accuracy_score

from kNN import kNN
import pandas as pd
from sklearn.model_selection import train_test_split
from visualize_clf import visualize_clf

from sklearn.preprocessing import MinMaxScaler


class kNN_test:
    def __init__(self):
        # Input Files
        self.dataset = pd.read_csv('virus_data.csv')

        # Splitting the data
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=123)

    def test_Q10(self):
        train_2X = self.train[['PCR_04', 'PCR_09']]
        train_Y = self.train[['spread']]
        knn1 = kNN(1)
        knn1.fit(train_2X, train_Y)
        test_2X = self.test[['PCR_04', 'PCR_09']]
        test_Y_real = self.test[['spread']]

        # Making our predictions
        test_Y_knn = knn1.predict(test_2X)

        # Output Scores and Graph
        test_score = accuracy_score(test_Y_real, test_Y_knn)
        print('Training score is: ' + str(knn1.score(train_2X, train_Y) * 100) + '%')
        print('Testing score is: ' + str(test_score * 100) + '%')
        visualize_clf(knn1, train_2X, train_Y, '1-NN Classification', 'PCR4', 'PCR9')

    def test_Q11(self):
        # Data Cleaning and Ordering.
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_2X_normal = scaler.fit_transform(self.train[['PCR_04', 'PCR_09']])
        train_Y = self.train[['spread']]
        test_2X_normal = scaler.transform(self.test[['PCR_04', 'PCR_09']])
        test_Y_real = self.test[['spread']]

        # Making our predictions
        knn1 = kNN(1)
        knn1.fit(train_2X_normal, train_Y)
        test_Y_knn = knn1.predict(test_2X_normal)

        # Output Scores and Graph
        test_score = accuracy_score(test_Y_real, test_Y_knn)
        print('Training score is: ' + str(knn1.score(train_2X_normal, train_Y) * 100) + '%')
        print('Testing score is: ' + str(test_score * 100) + '%')
        visualize_clf(knn1, train_2X_normal, train_Y, '1-NN Classification Normalized', 'PCR4', 'PCR9')


    def test_Q12(self):
        # Data Cleaning and Ordering.
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_2X_normal = scaler.fit_transform(self.train[['PCR_04', 'PCR_09']])
        train_Y = self.train[['spread']]
        test_2X_normal = scaler.transform(self.test[['PCR_04', 'PCR_09']])
        test_Y_real = self.test[['spread']]

        # Making our predictions
        knn5 = kNN(5)
        knn5.fit(train_2X_normal, train_Y)
        test_Y_knn = knn5.predict(test_2X_normal)

        # Output Scores and Graph
        test_score = accuracy_score(test_Y_real, test_Y_knn)
        print('Training score is: ' + str(knn5.score(train_2X_normal, train_Y) * 100) + '%')
        print('Testing score is: ' + str(test_score * 100) + '%')
        visualize_clf(knn5, train_2X_normal, train_Y, '5-NN Classification Normalized', 'PCR4', 'PCR9')

    def test_Q21(self):
        # # Data Cleaning and Ordering.
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # train_2X_normal = scaler.fit_transform(self.train[['PCR_03', 'PCR_10']])
        # train_Y = self.train[['risk']]
        # test_2X_normal = scaler.transform(self.test[['PCR_03', 'PCR_10']])
        # test_Y_real = self.test[['risk']]
        #
        # # Making our predictions
        # knn5 = kNN(10)
        # knn5.fit(train_2X_normal, train_Y)
        # test_Y_knn = knn5.predict(test_2X_normal)
        #
        # # Output Scores and Graph
        # test_score = accuracy_score(test_Y_real, test_Y_knn)
        # print('Training score is: ' + str(knn5.score(train_2X_normal, train_Y) * 100) + '%')
        # print('Testing score is: ' + str(test_score * 100) + '%')
        # visualize_clf(knn5, train_2X_normal, train_Y, '2-NN Classification Normalized', 'PCR4', 'PCR9')
        train_2X = self.train[['PCR_03', 'PCR_10']]
        train_Y = self.train[['risk']]
        knn1 = kNN(3)
        knn1.fit(train_2X, train_Y)
        test_2X = self.test[['PCR_03', 'PCR_10']]
        test_Y_real = self.test[['risk']]

        # Making our predictions
        test_Y_knn = knn1.predict(test_2X)

        # Output Scores and Graph
        test_score = accuracy_score(test_Y_real, test_Y_knn)
        print('Training score is: ' + str(knn1.score(train_2X, train_Y) * 100) + '%')
        print('Testing score is: ' + str(test_score * 100) + '%')
        visualize_clf(knn1, train_2X, train_Y, '1-NN Classification', 'PCR3', 'PCR10')








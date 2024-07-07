from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
from prettytable import PrettyTable

class GroupKFoldCV:
    def __init__(self, models, n_folds=4, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42, n_clusters=4):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.results_table_ = PrettyTable([date + ' GroupKFold CV', 'RMSE_AVG', 'R2_AVG'])

    def __run__(self):
        for name in self.models_:
            model = self.models_[name]

            clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.X_train).labels_

            gkf = GroupKFold(n_splits=self.n_folds)

            scores = cross_validate(model, self.X_train, self.y_train, return_estimator=True, scoring=self.scoring_, cv=gkf, groups=clusters)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']

            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)
            test_RMSE = root_mean_squared_error(self.y_test, predictions)
            test_R2 = model.score(self.X_test, self.y_test)
            
            self.results_table_.add_row([f'{name} CV Predicted', np.average(cv_RMSE_), np.average(cv_R2)])
            self.results_table_.add_row([f'{name} Test', test_RMSE, test_R2])

    def results(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.__run__()

        print(self.results_table_)


        
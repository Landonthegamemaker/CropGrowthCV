from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
from prettytable import PrettyTable 

class RandomCV:
    def __init__(self, models, n_folds=5, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state

        self.results_table_ = PrettyTable([date + ' Random CV', 'RMSE_AVG', 'R2_AVG'])

    def __run__(self):
        for name in self.models_:
            model = self.models_[name]
            cv_method = ShuffleSplit(n_splits=self.n_folds, test_size=self.test_size, random_state=self.random_state)
            scores = cross_validate(model, self.X_train, self.y_train, return_estimator=True, scoring=self.scoring_, cv=cv_method)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']

            test_scores = {'R2': [], 'RMSE': []}

            for estimator in scores['estimator']:
                predictions = estimator.predict(self.X_test)
                test_R2 = estimator.score(self.X_test, self.y_test)
                test_RMSE = root_mean_squared_error(self.y_test, predictions)

                test_scores['R2'].append(test_R2)
                test_scores['RMSE'].append(test_RMSE)
            
            self.results_table_.add_row([f'{name} CV Predicted', np.average(cv_RMSE_), np.average(cv_R2)])
            self.results_table_.add_row([f'{name} Test', np.average(test_scores['RMSE']), np.average(test_scores['R2'])])

    def results(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.__run__()

        print(self.results_table_)


        
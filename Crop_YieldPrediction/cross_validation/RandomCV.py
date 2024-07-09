from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np

class RandomCV:
    def __init__(self, models, n_folds=5, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42, method='LOFO'):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.method = method

        self.results_table = pd.DataFrame(columns=['MODEL', 'CV_METHOD', 'RMSE_AVG', 'R2_AVG'])

    def __run__(self):
        cv_method = ShuffleSplit(n_splits=self.n_folds, test_size=self.test_size, random_state=self.random_state)
        if self.method == 'LOFO':
            for name in self.models_:
                model = self.models_[name]
                scores = cross_validate(model, self.X_train, self.y_train, scoring=self.scoring_, cv=cv_method)
                cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
                cv_R2 = scores['test_r2']
                
                model.fit(self.X_train, self.y_train)

                predictions = model.predict(self.X_test)

                test_RMSE = root_mean_squared_error(self.y_test, predictions)
                test_R2 = model.score(self.X_test, self.y_test)

                self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'RNDM_CV_LOFO', np.average(cv_RMSE_), np.average(cv_R2)]
                self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'RNDM_TEST_LOFO', test_RMSE, test_R2]
        elif self.method == '3FIELD':
            # Hack to get around preprocessing class since it was implemented for LOFO
            X_combined = pd.concat([self.X_train, self.X_test])
            Y_combined = pd.concat([self.y_train, self.y_test])

            X_train, X_test, y_train, y_test = train_test_split(X_combined, Y_combined, test_size=0.2, random_state=self.random_state)

            for name in self.models_:
                model = self.models_[name]
                scores = cross_validate(model, X_train, y_train, scoring=self.scoring_, cv=cv_method)
                cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
                cv_R2 = scores['test_r2']

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                
                test_RMSE = root_mean_squared_error(y_test, predictions)
                test_R2 = model.score(X_test, y_test)

                self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'RNDM_CV_3_Field', np.average(cv_RMSE_), np.average(cv_R2)]
                self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'RNDM TEST_3_Field', test_RMSE, test_R2]
        else:
            raise ValueError("method " + self.method + "not found in [LOFO, 3FIELD]")
        

    def results(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.__run__()

        return self.results_table


        
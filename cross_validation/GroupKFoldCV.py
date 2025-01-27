from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class GroupKFoldCV:
    def __init__(self, models, n_folds=4, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42, n_clusters=4, method='LOFO'):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.method = method
        self.fold_results = {}

        self.results_table = pd.DataFrame(columns=['GroupKFold', 'CV_METHOD', 'RMSE_AVG', 'R2_AVG', 'RMSE STD', 'R2 STD'])

    def __run__(self):
        self.clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.X_train).labels_
        self.gkf = GroupKFold(n_splits=self.n_folds)
        for name in self.models_:
            model = self.models_[name]
            scores = cross_validate(model, self.X_train, self.y_train, return_estimator=True, scoring=self.scoring_, cv=self.gkf, groups=self.clusters)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']

            cv_RMSE_std = np.std(cv_RMSE_)
            cv_R2_std = np.std(cv_R2)

            self.fold_results[name] = {'RMSE': cv_RMSE_, 'R2': cv_R2}

            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)
            test_RMSE = root_mean_squared_error(self.y_test, predictions)
            test_R2 = model.score(self.X_test, self.y_test)
            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'GKF_CV_LOFO', np.average(cv_RMSE_), np.average(cv_R2), cv_RMSE_std, cv_R2_std]
            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'GKF_TEST_LOFO', test_RMSE, test_R2, 0, 0]
            
        

    def display_fold(self, fold_index: np.int32, train_coordinates: pd.DataFrame):
        splits = self.gkf.split(self.X_train, self.y_train, self.clusters)
        train_indices, test_indices = [list(traintest) for traintest in zip(*splits)]
        folds = [*zip(train_indices,test_indices)]

        fold_train_indices = folds[fold_index][0]
        fold_validate_indices = folds[fold_index][1]

        train = train_coordinates[np.in1d(train_coordinates.index, fold_train_indices)]
        validate = train_coordinates[np.in1d(train_coordinates.index, fold_validate_indices)]

        plt.scatter(train['X'], train['Y'], color='black', label='Train')
        plt.scatter(validate['X'], validate['Y'], color='red', label='Validate')
        plt.legend()
        plt.title("GKF Clustering Fold")
        plt.show()

        print('Train Samples   : ', len(fold_train_indices))
        print('Validate Samples: ', len(fold_validate_indices))
        print('LR Fold RMSE Scores: ', self.fold_results['LR']['RMSE'])
        print('LR Fold R2 Scores  : ', self.fold_results['LR']['R2'])
        print('RF Fold RMSE Scores: ', self.fold_results['RF']['RMSE'])
        print('RF Fold R2 Scores  : ', self.fold_results['RF']['R2'])

    def display_clusters(self, train_coordinates: pd.DataFrame):
        plt.scatter(train_coordinates['X'], train_coordinates['Y'], c=self.clusters)
        plt.title("GKF Clustering Results")
        plt.show()
    
    def results(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.__run__()

        return self.results_table


        
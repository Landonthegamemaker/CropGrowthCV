from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from util import ClusteringEnsamble
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display



class SpatialPlusCV:
    def __init__(self, models, n_folds=4, scoring = ['neg_root_mean_squared_error', 'r2'], date='', test_size=0.3, random_state=42, n_clusters=4):
        self.models_ = models
        self.n_folds = n_folds
        self.scoring_ = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.fold_results = {}

        self.results_table = pd.DataFrame(columns=['Spatial+', 'CV_METHOD', 'RMSE_AVG', 'R2_AVG', 'RMSE STD', 'R2 STD'])

    def __run__(self):
        clustering_labels = []

        # Get Clustering Ensamble Labels
        for col in self.X_train:
            labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.X_train[col].values.reshape(-1, 1)).labels_
            clustering_labels.append(np.array(labels))

        clustering_labels.append(KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.train_coordinates['X'].values.reshape(-1, 1)).labels_)
        clustering_labels.append(KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.train_coordinates['Y'].values.reshape(-1, 1)).labels_)

        ensamble = ClusteringEnsamble(n_clusters=self.n_clusters)
        self.ensamble_labels = ensamble.transform(clustering_labels)

        # Perform Cross Validation For Each Model
        self.gkf = LeaveOneGroupOut()        
        for name in self.models_:
            model = self.models_[name]

            scores = cross_validate(model, self.X_train, self.y_train, scoring=self.scoring_, cv=self.gkf, groups=self.ensamble_labels, return_estimator=True)
            cv_RMSE_ = -scores['test_neg_root_mean_squared_error']
            cv_R2 = scores['test_r2']
            cv_RMSE_std = np.std(cv_RMSE_)
            cv_R2_std = np.std(cv_R2)

            # Get Test Results for Comparison between CV Predicted and Actual
            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)
            test_RMSE = root_mean_squared_error(self.y_test, predictions)
            test_R2 = model.score(self.X_test, self.y_test)

            self.fold_results[name] = {'RMSE': cv_RMSE_, 'R2': cv_R2}

            # Add Results into table
            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'SP_CV_LOFO', np.average(cv_RMSE_), np.average(cv_R2), cv_RMSE_std, cv_R2_std]
            self.results_table.loc[len(self.results_table.index)] = [f'{name}', 'SP_TEST_LOFO', test_RMSE, test_R2, 0, 0]

    def display_fold(self, fold_index: np.int32):
        splits = self.gkf.split(self.X_train, self.y_train, self.ensamble_labels)
        train_indices, test_indices = [list(traintest) for traintest in zip(*splits)]
        folds = [*zip(train_indices,test_indices)]

        fold_train_indices = folds[fold_index][0]
        fold_validate_indices = folds[fold_index][1]

        train = self.train_coordinates[np.in1d(self.train_coordinates.index, fold_train_indices)]
        validate = self.train_coordinates[np.in1d(self.train_coordinates.index, fold_validate_indices)]

        plt.scatter(train['X'], train['Y'], color='black', label='Train')
        plt.scatter(validate['X'], validate['Y'], color='red', label='Validate')
        plt.legend()
        plt.title("SP Clustering Fold")
        plt.show()

        print('Train Samples   : ', len(fold_train_indices))
        print('Validate Samples: ', len(fold_validate_indices))
        print('LR Fold CV RMSE Scores: ', self.fold_results['LR']['RMSE'])
        print('LR Fold CV R2 Scores  : ', self.fold_results['LR']['R2'])
        print('RF Fold CV RMSE Scores: ', self.fold_results['RF']['RMSE'])
        print('RF Fold CV R2 Scores  : ', self.fold_results['RF']['R2'])

    def display_clusters(self, train_coordinates: pd.DataFrame):
        plt.scatter(train_coordinates['X'], train_coordinates['Y'], c=self.ensamble_labels)
        plt.title("SP Clustering Results")
        plt.show()

        
    def results(self, X_train, X_test, y_train, y_test, train_coordiantes):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_coordinates = train_coordiantes

        self.__run__()

        return self.results_table
        
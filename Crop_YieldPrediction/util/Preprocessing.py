import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import DataConversionWarning
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)

class CombinedPreprocessor:
    def __init__(self, drop=['Plot', 'Year', 'Date', 'Range', 'Row', 'left', 'top', 'right', 'X', 'Y', 'bottom', 'Mean.Yld.bu.ac', 'Yld Vol(Dr', 'Crop Flw(M'], inplace=False, test_size=0.3, target=['Yld Mass(D'], random_state=42):
        self.drop = drop
        self.inplace = inplace
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
    
    def __drop__(self):
        self.df1_.drop(columns=self.drop, axis='columns', inplace=True)
        self.df2_.drop(columns=self.drop, axis='columns', inplace=True)
        self.df3_.drop(columns=self.drop, axis='columns', inplace=True)

    def __scale__(self):
        self.scaler_ = MinMaxScaler()
        scalecols = [col for col in self.df1_.columns if col not in ['X', 'Y']]
        self.df1_[scalecols] = self.scaler_.fit_transform(self.df1_[scalecols])
        self.df2_[scalecols] = self.scaler_.transform(self.df2_[scalecols])
        self.df3_[scalecols] = self.scaler_.transform(self.df3_[scalecols])

    def transform(self, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
        for col in self.target:
            if col not in df1.columns:
                raise ValueError(f"Target Column {col} not found in {df1.columns}")
        self.df1_ = df1 if self.inplace else df1.copy(deep=True)
        self.df2_ = df2 if self.inplace else df2.copy(deep=True)
        self.df3_ = df3 if self.inplace else df3.copy(deep=True)
        self.__drop__()
        self.__scale__()

        xCols = [col for col in self.df1_.columns if col not in self.target]
        self.train_validate = pd.concat([self.df1_, self.df2_])

        self.X_train = self.train_validate[xCols]
        self.y_train = self.train_validate[self.target]

        self.X_test = self.df3_[xCols]
        self.y_test = self.df3_[self.target]

        # Hack to get spatial coordinates
        # self.train_coordinates = self.X_train[['X', 'Y']]
        # self.test_coordinates = self.X_test[['X', 'Y']]


        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform(self):
        self.df1_[self.df1_.columns] = self.scaler_.inverse_transform(self.df1_[self.df1_.columns])
        self.df2_[self.df1_.columns] = self.scaler_.inverse_transform(self.df2_[self.df1_.columns])
        self.df3_[self.df1_.columns] = self.scaler_.inverse_transform(self.df3_[self.df1_.columns])
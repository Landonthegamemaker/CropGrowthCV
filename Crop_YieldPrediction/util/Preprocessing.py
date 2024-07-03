import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning

import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)

class CombinedPreprocessor:
    def __init__(self, drop=['Plot', 'Year', 'Date', 'Range', 'Row', 'left', 'top', 'right', 'bottom', 'X', 'Y', 'Yld Vol(Dr', 'Crop Flw(M', 'Crop Flw(V'], inplace=False, test_size=0.3, target=['Yld Mass(D'], random_state=42):
        self.drop = drop
        self.inplace = inplace
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
    
    def __drop__(self):
        self.df1_.drop(columns=self.drop, axis='columns', inplace=True)
        self.df2_.drop(columns=self.drop, axis='columns', inplace=True)
        self.df3_.drop(columns=self.drop, axis='columns', inplace=True)
        self.df3_.drop(columns=['id_old'], axis='columns', inplace=True)

    def __scale__(self):
        self.scaler_ = MinMaxScaler()
        self.df1_[self.df1_.columns] = self.scaler_.fit_transform(self.df1_[self.df1_.columns])
        self.df2_[self.df2_.columns] = self.scaler_.transform(self.df2_[self.df2_.columns])
        self.df3_[self.df3_.columns] = self.scaler_.transform(self.df3_[self.df3_.columns])

    # def __split__(self):
    #     xCols = [col for col in self.df1_.columns if col not in self.target]
        
    #     X_train1, X_test1, y_train1, y_test1 = train_test_split(self.df1_[xCols], self.df1_[self.target], test_size=self.test_size, random_state=self.random_state)
    #     X_train2, X_test2, y_train2, y_test2 = train_test_split(self.df2_[xCols], self.df2_[self.target], test_size=self.test_size, random_state=self.random_state)

    #     self.X_train = pd.concat([X_train1, X_train2])
    #     self.X_valid = pd.concat([X_test1, X_test2])
    #     self.y_train = pd.concat([y_train1, y_train2])
    #     self.y_valid = pd.concat([y_test1, y_test2])

    #     self.X_test = self.df3[xCols]
    #     self.Y_test = self.df3[self.target]

    def transform(self, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
        for col in self.target:
            if col not in df1.columns:
                raise ValueError(f"Target Column {col} not found in {df1.columns}")
        self.df1_ = df1 if self.inplace else df1.copy(deep=True)
        self.df2_ = df2 if self.inplace else df2.copy(deep=True)
        self.df3_ = df3 if self.inplace else df3.copy(deep=True)
        self.__drop__()
        self.__scale__()
        # self.__split__()
        # return self.X_train, self.X_valid, self.y_train, self.y_valid, self.X_test, self.Y_test


        xCols = [col for col in self.df1_.columns if col not in self.target]
        self.train_validate = pd.concat([self.df1_, self.df2_])

        self.X_train = self.train_validate[xCols]
        self.y_train = self.train_validate[self.target]

        self.X_test = self.df3_[xCols]
        self.y_test = self.df3_[self.target]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform(self):
        self.df1_[self.df1_.columns] = self.scaler_.inverse_transform(self.df1_[self.df1_.columns])
        self.df2_[self.df2_.columns] = self.scaler_.inverse_transform(self.df2_[self.df2_.columns])
        self.df3_[self.df3_.columns] = self.scaler_.inverse_transform(self.df3_[self.df3_.columns])
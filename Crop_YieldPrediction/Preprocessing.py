import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CombinedPreprocessor:
    def __init__(self, drop=['Plot', 'Year', 'Date', 'Range', 'Row', 'left', 'top', 'right', 'bottom', 'X', 'Y', 'Yld Vol(Dr', 'Crop Flw(M', 'Crop Flw(V'], inplace=False, test_size=0.2, target=['Yld Mass(D'], random_state=42):
        self.drop = drop
        self.inplace = inplace
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
    
    def __drop__(self):
        self.df_.drop(columns=self.drop, axis='columns', inplace=True)

    def __scale__(self):
        self.scaler_ = MinMaxScaler()
        self.df_[self.df_.columns] = self.scaler_.fit_transform(self.df_[self.df_.columns])

    def __split__(self):
        xCols = [col for col in self.df_.columns if col not in self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_[xCols], self.df_[self.target], test_size=self.test_size, random_state=self.random_state)

    def transform(self, df: pd.DataFrame):
        for col in self.target:
            if col not in df.columns:
                raise ValueError(f"Target Column {col} not found in {df.columns}")
        self.df_ = df if self.inplace else df.copy(deep=True)
        self.target = self.target
        self.__drop__()
        self.__scale__()
        self.__split__()

        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform(self):
        self.df_[self.df_.columns] = self.scaler_.inverse_transform(self.df_[self.df_.columns])
import pandas as pd
import numpy as np

from util import CombinedPreprocessor
from cross_validation import RandomCV, GroupKFoldCV, SpatialPlusCV, MSpatialPlusCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(41)

print("6/18/2020, 6/16/2021, 6/15/2022")
print("-------------------------------")

indices_6182020 = pd.read_csv("../Datasets/Crop_Yield/Indices_Combined/2020/June_18_2020_New.csv")
indices_6162021 = pd.read_csv("../Datasets/Crop_Yield/Indices_Combined/2021/June_16_2021.csv")
indices_6152022 = pd.read_csv("../Datasets/Crop_Yield/Indices_Combined/2022/June_15_2022.csv")
processor = CombinedPreprocessor()

X_train, X_test, y_train, y_test = processor.transform(indices_6162021, indices_6152022, indices_6182020)

random_validator = RandomCV(models={'LR': LinearRegression(n_jobs=-1), 'RF': RandomForestRegressor(n_jobs=-1, random_state=41)})
gkf_validator = GroupKFoldCV(models={'LR': LinearRegression(n_jobs=-1), 'RF': RandomForestRegressor(n_jobs=-1, random_state=41)})
sp_validator = SpatialPlusCV(models={'LR': LinearRegression(n_jobs=-1), 'RF': RandomForestRegressor(n_jobs=-1, random_state=41)})
msp_validator = MSpatialPlusCV(models={'LR': LinearRegression(n_jobs=-1), 'RF': RandomForestRegressor(n_jobs=-1, random_state=41)})

random_validator.results(X_train, X_test, y_train, y_test)
gkf_validator.results(X_train, X_test, y_train, y_test)
sp_validator.results(X_train, X_test, y_train, y_test)
msp_validator.results(X_train, X_test, y_train, y_test)
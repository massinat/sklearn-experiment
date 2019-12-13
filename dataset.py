"""
This class represents the dataset
Contains method to initialize, manipulate and filter all the data loaded from the input file

@Author:        Massimiliano Natale
@Student id:        
"""

import pandas as pd
from sklearn import preprocessing


class Dataset:
    def __init__(self, inputFile):
        self._originalDataset = pd.read_csv(inputFile, delimiter=",").drop("id", axis=1).drop("date", axis=1)
        self._workingDataset = self._originalDataset.copy()
        self._X = self._workingDataset.drop("price", axis=1)
        self._y = self._workingDataset["price"]

    @property
    def data(self):
        return self._workingDataset

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def columns(self):
        return self.data.columns

    def info(self):
        print(self.data.head())
        print(self.data.info())
        for item in self.data.columns:
            print(f"{item}:{self.data[item].nunique()}")

    def addCompositeAttribute(self, name, lambdaCalculation):
        if name in self.data.columns:
            raise ValueError(f"The dataset already contains the property {name}")

        self.data[name] = lambdaCalculation()

    def normalizeData(self):
        normalizer = preprocessing.MinMaxScaler()
        self._X = normalizer.fit_transform(self._X)

    def standardizeData(self):
        standardizer = preprocessing.StandardScaler()
        self._X = standardizer.fit_transform(self._X)

    def restoreDataset(self):
        self._workingDataset = self._originalDataset.copy()
        self._X = self._workingDataset.drop("price", axis=1)
        self._y = self._workingDataset["price"]

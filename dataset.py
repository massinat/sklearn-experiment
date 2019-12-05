"""
This class represents the dataset
Contains method to initialize, manipulate and filter all the data loaded from the input file

@Author:        Massimiliano Natale
@Student id:        
"""

import pandas as pd

class Dataset:

    def __init__(self, inputFile):
        self._originalDataset = pd.read_csv(inputFile, delimiter=",")
        self.workingDataset = self._originalDataset.copy()
        self._X = self.dataset.drop("price", axis=1)
        self._y = self.dataset["price"]

    @property
    def dataset(self):
        return self._originalDataset

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self.y

    def filterByAttributesNames(self, *names):
        return self.workingDataset[names]

    def addCompositeAttribute(self, name, lambdaCalculation):
        if name in self.workingDataset.columns:
            raise ValueError(f"The dataset already contains the property {name}")

        self.workingDataset[name] = lambdaCalculation()

    def restoreDataset(self):
        self.workingDataset = self.dataset.copy()
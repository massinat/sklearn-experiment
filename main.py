import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

class Solution:

    def __init__(self, inputFile, outputFile):
        self._inputFile = inputFile
        self.dataset = pd.read_csv(inputFile, delimiter=",")
        self.X = self.dataset.drop("price", axis=1)
        self.y = self.dataset["price"]
        self.outputFile = outputFile

    # Let's nicely explore our data to gain more insight
    def exploreData(self):
        print(self.dataset.head())
        self.dataset.info()
        print(self.dataset["waterfront"].value_counts())
        print(self.dataset["view"].value_counts())
        print(self.dataset["condition"].value_counts())
        print(self.dataset["grade"].value_counts())

        # self.X.hist(bins=50, figsize=(20,15))
        # pyplot.show()

        # self.X.plot(kind="scatter", x="long", y="lat")
        # pyplot.show()

        # self.X.plot(kind="scatter", x="long", y="lat", alpha=0.1)
        # pyplot.show()

        #self.dataset.plot(kind="scatter", x="long", y="lat", alpha=0.4, s=self.dataset["sqft_living"]/100, label="Living space", figsize=(10,7), c="price", cmap=pyplot.get_cmap("jet"), colorbar=True)
        #pyplot.show()

        # correlationMatrix = self.dataset.corr()
        # print(correlationMatrix["price"].sort_values(ascending=False))

        attributes = ["price", "sqft_living", "grade"] # to be continued
        scatter_matrix(self.dataset[attributes], figsize=(12,8))
        pyplot.show()

if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.exploreData()

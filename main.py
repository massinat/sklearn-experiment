import numpy as np
import pandas as pd
from matplotlib import pyplot

class Solution:

    def __init__(self, inputFile, outputFile):
        self._inputFile = inputFile
        self.dataset = pd.read_csv(inputFile, delimiter=",")
        self.X = self.dataset.drop("price", axis=1)
        self.y = self.dataset["price"]
        self.outputFile = outputFile

    # Let's nicely explore our data to gain more insight
    def exploreData(self):
        self.dataset.hist()
        pyplot.show()

if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.exploreData()

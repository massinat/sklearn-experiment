"""
Main class
It executes the whole experiment: data visualization, elaboration and model

@Author:        Massimiliano Natale
@Student id:        
"""

from dataset import Dataset
from dataVisualizer import DataVisualizer


class Solution:

    def __init__(self, inputFile, outputFile):
        self.dataset = Dataset(inputFile)
        self.dataVisualizer = DataVisualizer()
        self._outputFile = outputFile

    # Let's nicely explore our data to gain more insight
    def exploreData(self):
        ##!self.dataset.info()

        ##!for item in [x for x in self.dataset.columns if x != "date"]:
            ##!self.dataVisualizer.showHistogram(self.dataset.data[[item]])

        ##!self.dataVisualizer.showScatter(self.dataset.X, "long", "lat")
        ##!self.dataVisualizer.showScatter(self.dataset.X, "long", "lat", 0.1)

        ##!self.dataVisualizer.showHeatMap(self.dataset.data, "long", "lat", "Living space", "sqft_living", "price", lambda x: x/100)
        ##!self.dataVisualizer.showHeatMap(self.dataset.data[self.dataset.data["price"] <= 500000], "long", "lat", "Living space", "sqft_living", "price", lambda x: x/100)

        # correlationMatrix = self.dataset.corr()
        # print(correlationMatrix["price"].sort_values(ascending=False))

        # attributes = ["price", "sqft_living", "grade"] # to be continued
        # scatter_matrix(self.dataset[attributes], figsize=(12,8))
        # pyplot.show()

if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.exploreData()

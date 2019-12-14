"""
Main class
It executes the whole experiment: data visualization, elaboration and model

@Author:        Massimiliano Natale
@Student id:        
"""

from dataset import Dataset
from modelBuilder import ModelBuilder
from dataVisualizer import DataVisualizer


class Solution:

    def __init__(self, inputFile, outputFile):
        self.dataset = Dataset(inputFile)
        self.modelBuilder = ModelBuilder()
        self.dataVisualizer = DataVisualizer()
        self._outputFile = outputFile

    # Let's nicely explore our data to gain more insight
    def exploreData(self):
        ##!self.dataset.info()

        # Data distribution and univariate outlier detection
        ##!for item in [x for x in self.dataset.columns]:
            ##!self.dataVisualizer.showHistogram(self.dataset.data[[item]])
            ##!self.dataVisualizer.showBoxplot(self.dataset.data[[item]])

        # Ordinal categorical feature in our dataset [No categorical feature]

        ##!self.dataVisualizer.showScatter(self.dataset.X, "long", "lat")
        ##!self.dataVisualizer.showScatter(self.dataset.X, "long", "lat", 0.1)

        ##!self.dataVisualizer.showHeatMap(self.dataset.data, "long", "lat", "Living space", "sqft_living", "price", lambda x: x/100)
        ##!self.dataVisualizer.showHeatMap(self.dataset.data[self.dataset.data["price"] <= 500000], "long", "lat", "Living space", "sqft_living", "price", lambda x: x/100)

        # Having removed the outliers, we choose for normalising data. We can use both and decide what works better [suggested approach]
        ##!self.dataset.normalizeData()

        # Memory consumption for tree based feature selection. We cut the number of instances
        ##!self.modelBuilder.univariateFeatureSelection(self.dataset.X, self.dataset.y, 30)
        ##!self.modelBuilder.treeBasedFeatureSelection(self.dataset.X, self.dataset.y, 1, 20000)
        self.modelBuilder.greedyFeaturSelection(self.dataset.X, self.dataset.y)

if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.exploreData()

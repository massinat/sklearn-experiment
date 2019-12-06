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
        self.dataset.info()

        # self.dataVisualizer.showHistogram(self.dataset.filterByAttributesNames("price", "bedrooms", "bathrooms", "sqft_living"), 50)
        # self.dataVisualizer.showHistogram(self.dataset.filterByAttributesNames("sqft_lot", "floors", "waterfront", "view"), 50)
        # self.dataVisualizer.showHistogram(self.dataset.filterByAttributesNames("condition", "grade", "sqft_above", "sqft_basement"), 50)
        # self.dataVisualizer.showHistogram(self.dataset.filterByAttributesNames("yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"), 50)

        self.dataVisualizer.showScatter(self.dataset.X, "long", "lat")
        self.dataVisualizer.showScatter(self.dataset.X, "long", "lat", 0.1)

        #self.dataset.plot(kind="scatter", x="long", y="lat", alpha=0.4, s=self.dataset["sqft_living"]/100, label="Living space", figsize=(10,7), c="price", cmap=pyplot.get_cmap("jet"), colorbar=True)
        #pyplot.show()

        # correlationMatrix = self.dataset.corr()
        # print(correlationMatrix["price"].sort_values(ascending=False))

        # attributes = ["price", "sqft_living", "grade"] # to be continued
        # scatter_matrix(self.dataset[attributes], figsize=(12,8))
        # pyplot.show()

if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.exploreData()

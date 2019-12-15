"""
Main class
It executes the whole experiment: data visualization, elaboration and model

@Author:        Massimiliano Natale
@Student id:        
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dataset import Dataset
from modelBuilder import ModelBuilder
from dataVisualizer import DataVisualizer


class Solution:
    def __init__(self, inputFile, outputFile):
        self.dataset = Dataset(inputFile)
        self.modelBuilder = ModelBuilder()
        self.dataVisualizer = DataVisualizer()
        self._outputFile = outputFile

    # Data engineering
    def engineerData(self):
        self.dataset.transformAttribute("date", lambda x: pd.to_datetime(x))
        self.dataset.transformAttribute("price", lambda x: x.astype(int))
        self.dataset.transformAttribute("bathrooms", lambda x: x.astype(int))
        self.dataset.transformAttribute("floors", lambda x: x.astype(int))
        self.dataset.addCompositeAttribute(
            "houseAge", "date", "yr_built", lambda x, y: x.dt.year - y
        )
        self.dataset.addAttribute(
            "renovated", "yr_renovated", lambda x: 0 if x == 0 else 1
        )

        self.dataset.removeAttribute("date")
        self.dataset.removeAttribute("yr_renovated")
        self.dataset.removeAttribute("yr_built")

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

        # Memory consumption for tree based feature selection. We cut the number of instances
        ##!self.modelBuilder.univariateFeatureSelection(self.dataset.X, self.dataset.y, 30)
        ##!self.modelBuilder.treeBasedFeatureSelection(self.dataset.X, self.dataset.y, 1, 20000)
        ##!self.modelBuilder.greedyFeatureSelection(self.dataset.X, self.dataset.y)

        # Split training and testa data here?
        ##!self.dataset.splitTrainingTestData()

        ##!self.modelBuilder.addToPipeline("Linear Pipeline", StandardScaler, "Linear Regression", LinearRegression)
        ##!self.modelBuilder.addToPipeline("KNN Pipeline", StandardScaler, "KNN", KNeighborsRegressor)
        ##!self.modelBuilder.addToPipeline("Bayesian Pipeline", StandardScaler, "Bayesian Ridge", BayesianRidge)
        ##!self.modelBuilder.addToPipeline("Elastic Pipeline", StandardScaler, "Elastic", ElasticNet)
        ##!self.modelBuilder.addToPipeline("Decision Pipeline", StandardScaler, "Decision Tree", DecisionTreeRegressor)
        ##!self.modelBuilder.addToPipeline("Gradient", StandardScaler, "Gradient", GradientBoostingRegressor)

        ##!self.modelBuilder.evaluateModels(self.dataset.XTrain, self.dataset.yTrain)

        # self.modelBuilder.searchBestHyperparameters(
        #     self.dataset.XTrain,
        #     self.dataset.yTrain,
        #     StandardScaler,
        #     "KNN",
        #     KNeighborsRegressor,
        #     dict(
        #         n_neighbors=np.array(
        #             [1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 150, 300, 500, 1000]
        #         ),
        #         weights=np.array(["uniform", "distance"]),
        #         algorithm=np.array(["auto", "ball_tree", "kd_tree", "brute"]),
        #         p=np.array([1, 2, 3]),
        #     ),
        # )

        self.modelBuilder.searchBestHyperparameters(
            self.dataset.XTrain,
            self.dataset.yTrain,
            StandardScaler,
            "Gradient",
            GradientBoostingRegressor,
            dict(
                random_state=np.array([42]),
                n_estimators=np.array(
                    [50, 100, 200, 300, 400, 500, 1000, 1500, 2500, 5000]
                ),
                loss=np.array(["ls", "lad", "huber", "quantile"]),
            ),
        )


if __name__ == "__main__":
    solution = Solution("housing.csv", "output.txt")
    solution.dataset.info()
    solution.engineerData()
    solution.dataset.info()

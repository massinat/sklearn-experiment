"""
Core classification class
It performs the main steps in order to build the model

@Author:        Massimiliano Natale
@Student id:        
"""

from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.ensemble import RandomForestClassifier


class ModelBuilder:

    def univariateFeatureSelection(self, X, y, featureNames, percentile=10):
        selector = SelectPercentile(f_regression, percentile=percentile)
        selector.fit(X, y)

        for featureName, score in zip(featureNames, selector.scores_):
            print(f"{featureName}={score}")

    def treeBasedFeatureSelection(self, X, y, nEstimators):
        forest = RandomForestClassifier(n_estimators=nEstimators, random_state=42)
        forest.fit(X, y)

        importances = forest.feature_importances_

        for index in importances:
            print(f"Feature[{index}]={importances[index]}")
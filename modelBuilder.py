"""
Core classification class
It performs the main steps in order to build the model

@Author:        Massimiliano Natale
@Student id:        
"""

from sklearn.feature_selection import SelectPercentile, f_regression, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


class ModelBuilder:

    def univariateFeatureSelection(self, X, y, percentile=20):
        selector = SelectPercentile(f_regression, percentile=percentile)
        selector.fit(X, y)

        for featureName, score in sorted(zip(X.columns, selector.scores_), key=lambda x: x[1], reverse=True):
            print(f"{featureName}={score}")

    def treeBasedFeatureSelection(self, X, y, nEstimators, nInstances):
        forest = RandomForestClassifier(n_estimators=nEstimators, random_state=42, n_jobs=-1)
        forest.fit(X.sample(n=nInstances, random_state=42), y.sample(n=nInstances, random_state=42))

        importances = forest.feature_importances_

        result = []
        for index, featureName in enumerate(X.columns):
            result.append((featureName, importances[index]))

        for item in sorted(result, key=lambda x: x[1], reverse=True):
            print(f"{item[0]}={item[1]}")

    def greedyFeaturSelection(self, X, y):
        estimator = linear_model.LogisticRegression(multi_class="auto", solver="lbfgs")
        rfecv = RFECV(estimator, cv=15)
        rfecv.fit(X, y)

        print(rfecv.n_features_)
        print(rfecv.ranking_)
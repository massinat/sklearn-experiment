"""
This class will visualize all the charts used for this assignment

@Author:        Massimiliano Natale
@Student id:  
"""

import pandas as pd
import matplotlib.pyplot as plt


class DataVisualizer:

    def showScatter(self, dataset, xName, yName, alpha=1):        
        dataset.plot(kind="scatter", x=xName, y=yName, alpha=alpha)
        plt.show()

    def showHistogram(self, dataset, bins):
        dataset.hist(bins=bins)
        plt.show()

    def showHeatMap(self, dataset, xName, yName, label, radiusAttribute, colorAttribute, alpha=1):
        dataset.plot(kind="scatter", x=xName, y=yName, label=label, s=radiusAttribute, c=colorAttribute, cmap=plt.get_cmap("jet"), alpha=alpha, colorbar=True)
        plt.show()

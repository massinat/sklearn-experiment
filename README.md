# sklearn-experiment
Scikit-Learn experiment

Let's start by taking look to the data self.dataset.head()
Let'have a quick description of the data, in particular the total number of rows, each attribute's type and the number of nonull values (important) self.dataset.info()
We see we don't have null values then no imputator needed.

All data all numerical except for the date (that is an an object). Maybe will we remove it? Let's do value_counts() anyway.
A better quick way to get a feel of the type of data we are dealing with is to plot a histogram for wach numerical attribute. A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis). We can plot one attribute at a time or we can use the hist() method on the whole dataset:
hist()

Let's have a visual of the map with self.X.plot(kind="scatter", x="long", y="lat"), it seems the map. Anyway let's use alpha to have a better insight (we visualize the place the higher density).
It's see the density near the see.

Let's create the heat map now to see how the prices change depending by the position. x.plot() etc etc

Let's compute the standard correlationMatrix to see how the values are linearly correlated. Let's use the scatter matrix to have a visual ones of the most promising values.

Let's visualize the outliers, this impacted the previous heat map due to price outliers. Can we create new attributes to better consider all the square metres? Think about it.

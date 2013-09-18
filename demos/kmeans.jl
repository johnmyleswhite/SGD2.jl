using SGD2
using RDatasets

iris = data("datasets", "iris")

X = matrix(iris[:, 2:5])'

m = KMeansSGD(X, 3)

fit!(m, X)

m

using SGD2
using RDatasets

iris = data("datasets", "iris")

X = matrix(iris[:, 2:5])'

p, n = size(X)

y = [s == "setosa" ? +1.0 : -1.0 for s in iris["Species"]]

m = SVMSGD(p,
	       n,
	       adagrad = true,
	       polyak = true,
	       eta = 0.01)

fit!(m, X, y)

cost(m)

module SGD2
	using Distributions
	using NumericExtensions
	using Distance

	invlogit(z::Real) = 1 / (1 + exp(-z))

	export LinearSGD, LogisticSGD, LassoSGD, SVMSGD, KMeansSGD
	export cost, predict!, residuals!, update!, fit!

	include("types.jl")
	include("predict.jl")
	include("residuals.jl")
	include("gradient.jl")
	include("cost.jl")
	include("update.jl")
	include("fit.jl")
	include("kmeans.jl")
end

type KMeansSGD
	n::Vector{Int}
	W::Matrix{Float64}
	d::Vector{Float64} # Buffer for distances from centers
	# TODO: Store assignments?
end

function KMeansSGD(X::Matrix, k::Integer = 2)
	# Need to be very careful about initialization
	# TODO: Try to use k-means++
	p, n = size(X)
	W = zeros(p, k)
	for j in 1:k
		i = rand(1:n)
		W[:, j] = X[:, i]
	end
	KMeansSGD(zeros(Int, k), W, zeros(k))
end

function fit!(m::KMeansSGD, X::Matrix, passes::Integer = 1)
	p, n = size(X)
	p, K = size(m.W)
	# TODO: Can multiple passes be tolerated?
	# Resetting all the values in W isn't a good thing
	for itr in 1:passes
		fill!(m.n, 1)
		for j in 1:n
			for k in 1:K
				m.d[k] = euclidean(X[:, j], m.W[:, k])
			end
			k_star = indmin(m.d)
			m.n[k_star] += 1
			m.W[:, k_star] = m.W[:, k_star] + (1 / m.n[k_star]) * (X[:, j] - m.W[:, k_star])
		end
	end
	return
end

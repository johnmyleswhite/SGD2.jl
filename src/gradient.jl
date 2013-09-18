# Note: These are all the negative gradient
function gradient!(m::Union(LinearSGD, LogisticSGD),
	               X::Matrix,
	               y::Vector)
	fill!(m.gr, 0.0)

	residuals!(m, X, y)

	# Log likelihood gradient
	p, n = size(X)
	b_gr = 0.0
	for j in 1:n
		b_gr += m.r[j]
		for i in 1:p
			m.gr[i] += m.r[j] * X[i, j]
		end
	end

	# Apply regularizer here
	for i in 1:p
		m.gr[i] += m.λ * -m.w[i]
	end

	return b_gr
end

function gradient!(m::SVMSGD,
	               X::Matrix,
	               y::Vector)
	fill!(m.gr, 0.0)

	residuals!(m, X, y)

	p, n = size(X)

	for j in 1:n
		if m.r[j] < 1
			for i in 1:p
				m.gr[i] += y[j] * X[i, j]
			end
		end
	end

	# Apply regularizer here
	for i in 1:p
		m.gr[i] += m.λ * -m.w[i]
	end

	return 0.0
end

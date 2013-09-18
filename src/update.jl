function update!(m::SGDModel, X::Matrix, y::Vector)
	p, n = size(X)

	if length(y) != n
		error("X and y do not match")
	end

	m.n += n

	b_gr = gradient!(m, X, y)

	α = m.lr(m)

	if m.adagrad
		if m.intercept
			m.s_b += b_gr^2
			m.b += α * b_gr / (m.τ0 + sqrt(m.s_b))
		end
		for i in 1:p
			m.s[i] += m.gr[i]^2
			# NB: This share τ0 between AdaGrad and 
			#     learning rate calculation
			m.w[i] += α * m.gr[i] / (m.τ0 + sqrt(m.s[i]))
		end
	else
		if m.intercept
			m.b += α * b_gr
		end
		for i in 1:p
			m.w[i] += α * m.gr[i]
		end
	end

	# Only after 1 epoch
	# Need to "reset" n
	if m.polyak && m.n > n
		for i in 1:p
			m.z[i] = ((m.n - n) / m.n) * m.z[i] + (n / m.n) * m.w[i]
		end
	end
end

# TODO: Move this into code above
function update!(m::LassoSGD, X::Matrix, y::Vector)
	p, n = size(X)

	# TODO: Do residuals need to be recalculated per observation?
	for j in 1:n
		m.p[j] = dot(X[:, j], m.w)
		m.p[j] += m.b
		m.r[j] = y[j] - m.p[j]
		for i in 1:p
			m.u[i] = max(0, m.u[i] - m.η * (m.λ - m.r[j] * X[i, j]))
			m.v[i] = max(0, m.v[i] - m.η * (m.λ + m.r[j] * X[i, j]))
			m.w[i] = m.u[i] - m.v[i]
		end
		m.b += m.η * m.r[j]
	end

	# Need AdaGrad, Polyak, ...
	return
end

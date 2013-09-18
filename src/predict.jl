function predict!(m::Union(LinearSGD, LassoSGD),
	              X::Matrix,
	              y::Vector)
	At_mul_B(m.p, X, m.w)
	p, n = size(X)
	if m.intercept
		for i in 1:n
			m.p[i] += m.b
		end
	end
	return
end

function predict!(m::LogisticSGD,
	              X::Matrix,
	              y::Vector)
	At_mul_B(m.p, X, m.w)
	p, n = size(X)
	if m.intercept
		for i in 1:n
			m.p[i] += m.b
		end
	end
	for j in 1:n
		m.p[j] = invlogit(m.p[j])
	end
	return
end

function predict!(m::SVMSGD,
	              X::Matrix,
	              y::Vector)
	At_mul_B(m.p, X, m.w)
	return
end

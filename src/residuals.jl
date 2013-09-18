function residuals!(m::Union(LinearSGD, LogisticSGD, LassoSGD),
	                X::Matrix,
	                y::Vector)
	p, n = size(X)
	predict!(m, X, y)
	for j in 1:n
		m.r[j] = (y[j] - m.p[j])
	end
	return
end

function residuals!(m::SVMSGD,
	                X::Matrix,
	                y::Vector)
	p, n = size(X)
	predict!(m, X, y)
	for j in 1:n
		m.r[j] = y[j] * m.p[j]
	end
	return
end

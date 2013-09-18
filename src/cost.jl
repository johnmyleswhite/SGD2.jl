# All functions assume residuals are current

cost(m::LinearSGD) = m.λ * norm(m.w, 2) + 0.5 * sqsum(m.r)

cost(m::LassoSGD) = m.λ * norm(m.w, 1) + 0.5 * sqsum(m.r)

function cost(m::LogisticSGD)
	n = length(m.p)

	ll = 0.0

	for i in 1:n
		if m.r[i] > 0.0
			y_i = 1.0
		else
			y_i = 0.0
		end
		ll += y_i * log(m.p[i]) + (1 - y_i) * log(1 - m.p[i])
	end

	return m.λ * norm(m.w, 2) - 2 * ll
end

function cost(m::SVMSGD)
	n = length(m.r)

	err = 0.0

	for i in 1:n
		err += max(0, 1 - m.r[i])
	end

	return m.λ * norm(m.w, 2) + err
end

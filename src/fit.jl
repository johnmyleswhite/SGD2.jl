function fit!(m::SGDModel,
	          X::Matrix,
	          y::Vector,
	          passes::Integer = 10_000)
	for pass in 1:passes
		update!(m, X, y)
	end
	return
end

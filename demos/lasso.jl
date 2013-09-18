using SGD2

srand(1)

p = 2
n = 1_000

X = randn(p, n)
β = randn(p)
y = X'β + randn(n)

m0 = LassoSGD(p,
	          n,
	          adagrad = true,
	          polyak = true,
	          eta = 0.001)

m1 = LassoSGD(p,
	          n,
	          adagrad = true,
	          polyak = true,
	          eta = 0.01)

m2 = LassoSGD(p,
	          n,
	          adagrad = true,
	          polyak = true,
	          eta = 0.1)

m3 = LassoSGD(p,
	          n,
	          adagrad = true,
	          polyak = true,
	          eta = 1.0)

i = -1
for m in [m0, m1, m2, m3]
	i += 1
	fit!(m, X, y)

	@printf "m%d - b: %s\n" i m.b
	@printf "m%d - w: %s\n" i join(m.w, ",")
	@printf "m%d - z: %s\n" i join(m.z, ",")
	@printf "m%d - β: %s\n" i join(β, ",")

	if m.polyak
		copy!(m.w, m.z)
	end
	residuals!(m, X, y)
	c1 = cost(m)

	m.b = 0.0
	copy!(m.w, β)
	residuals!(m, X, y)
	c2 = cost(m)

	@printf "m%d - Fit Cost: %f\n" i c1
	@printf "m%d - True Cost: %f\n" i c2
	@printf "\n"
end

using SGD2
using Distributions

srand(1)

p = 2
n = 1_000

m = zeros(p)
S = [10.0 0.9;
     0.9  1.0]
X = rand(MvNormal(m, S), n)
β = randn(p)
y = X'β
for i in 1:n
	y[i] = rand(Bernoulli(SGD2.invlogit(y[i])))
end

m1 = LogisticSGD(p,
	             n,
	             adagrad = true,
	             polyak = true,
	             eta = 0.1)

m2 = LogisticSGD(p,
	             n,
	             adagrad = false,
	             polyak = true,
	             eta = 0.1)

m3 = LogisticSGD(p,
	             n,
	             adagrad = true,
	             polyak = false,
	             eta = 0.1)

m4 = LogisticSGD(p,
	             n,
	             adagrad = true,
	             polyak = false,
	             eta = 0.1,
	             lambda = 100.0)

i = 0
for m in [m1, m2, m3, m4]
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

	copy!(m.w, β)
	residuals!(m, X, y)
	c2 = cost(m)

	@printf "m%d - Fit Cost: %f\n" i c1
	@printf "m%d - True Cost: %f\n" i c2
	@printf "\n"
end

abstract SGDModel

lr_constant(m::SGDModel) = m.η
lr_optimal(m::SGDModel) = m.η / (m.τ0 + m.n)
lr_root(m::SGDModel) = m.η / (m.n)^m.power

for ty in [:LinearSGD, :LogisticSGD, :SVMSGD]
	@eval begin
		type ($ty) <: SGDModel
			b::Float64          # Intercept
			s_b::Float64        # Adagrad squared gradient for intercept
			n::Int              # Number of observations seen
			w::Vector{Float64}  # Weights
			z::Vector{Float64}  # Polyak weights
			r::Vector{Float64}  # Residuals
			p::Vector{Float64}  # Predictions
			gr::Vector{Float64} # Gradient
			s::Vector{Float64}  # Adagrad squared gradient entries
			η::Float64          # Learning rate
			τ0::Float64         # A priori observation count
			power::Float64      # Learning rate
			λ::Float64          # L2 regularizer
			adagrad::Bool       # Use Adagrad rule?
			polyak::Bool        # Use Polyak-Ribbert averaging?
			intercept::Bool     # Use an intercept term?
			lr::Function        # Learning rate
		end
	end
end

type LassoSGD <: SGDModel
	b::Float64          # Intercept
	s_b::Float64        # Adagrad squared gradient for intercept
	n::Int              # Number of observations seen
	w::Vector{Float64}  # Weights
	u::Vector{Float64}  # "Latent" variables
	v::Vector{Float64}  # "Latent" variables
	z::Vector{Float64}  # Polyak weights
	r::Vector{Float64}  # Residuals
	p::Vector{Float64}  # Predictions
	gr::Vector{Float64} # Gradient
	s::Vector{Float64}  # Adagrad squared gradient entries
	η::Float64          # Learning rate
	τ0::Float64         # A priori observation count
	power::Float64      # Learning rate
	λ::Float64          # L1 regularizer
	adagrad::Bool       # Use Adagrad rule?
	polyak::Bool        # Use Polyak-Ribbert averaging?
	intercept::Bool     # Use an intercept term?
	lr::Function        # Learning rate
end

for ty in [:LinearSGD, :LogisticSGD]
	@eval begin
		function ($ty)(p::Integer,
			           n::Integer;
			           eta::Real = 0.01,
		               tau::Real = 0.0,
			           power::Real = 0.5,
			           lambda::Real = 0.0,
			           adagrad::Bool = true,
			           polyak::Bool = true,
			           intercept::Bool = true,
			           learn::Symbol = :constant)
			if learn == :optimal
				lr = lr_optimal
			elseif learn == :constant
				lr = lr_constant
			elseif learn == :root
				lr = lr_root
			else
				error("Unknown learning rate function")
			end

			($ty)(0.0,
				  0.0,
			      0,
			      zeros(p),
			      zeros(p),
			      zeros(n),
			      zeros(n),
			      zeros(p),
			      zeros(p),
			      float64(eta),
			      float64(tau),
			      float64(power),
			      float64(lambda),
			      adagrad,
			      polyak,
			      intercept,
			      lr)
		end
	end
end

for ty in [:SVMSGD]
	@eval begin
		function ($ty)(p::Integer,
			           n::Integer;
			           eta::Real = 0.01,
		               tau::Real = 0.0,
			           power::Real = 0.5,
			           lambda::Real = 0.0,
			           adagrad::Bool = true,
			           polyak::Bool = true,
			           intercept::Bool = false,
			           learn::Symbol = :constant)
			if learn == :optimal
				lr = lr_optimal
			elseif learn == :constant
				lr = lr_constant
			elseif learn == :root
				lr = lr_root
			else
				error("Unknown learning rate function")
			end

			($ty)(0.0,
				  0.0,
			      0,
			      zeros(p),
			      zeros(p),
			      zeros(n),
			      zeros(n),
			      zeros(p),
			      zeros(p),
			      float64(eta),
			      float64(tau),
			      float64(power),
			      float64(lambda),
			      adagrad,
			      polyak,
			      intercept,
			      lr)
		end
	end
end

function LassoSGD(p::Integer,
	              n::Integer;
	              eta::Real = 0.01,
                  tau::Real = 0.0,
	              power::Real = 0.5,
	              lambda::Real = 0.0,
	              adagrad::Bool = true,
	              polyak::Bool = true,
	              intercept::Bool = true,
	              learn::Symbol = :constant)
	if learn == :optimal
		lr = lr_optimal
	elseif learn == :constant
		lr = lr_constant
	elseif learn == :root
		lr = lr_root
	else
		error("Unknown learning rate function")
	end

	LassoSGD(0.0,
		     0.0,
	         0,
	         zeros(p),
	         zeros(p),
	         zeros(p),
	         zeros(p),
	         zeros(n),
	         zeros(n),
	         zeros(p),
	         zeros(p),
	         float64(eta),
	         float64(tau),
	         float64(power),
	         float64(lambda),
	         adagrad,
	         polyak,
	         intercept,
	         lr)
end

using LinearAlgebra



mutable struct corrmap
	δ::Vector{T} where T <: Number
	D::Int64
	De::Int64
	index::Array{T, 2} where T <: Number
	aval::T where T <: Number
	realtoz::Function
	ztoreal::Function
	rtoreal::Function
	realtor::Function
end

function corrmap(δ::Vector{T}, D::Int64) where {T <: Number}
	aval = 1

	De = Int64(D*(D-1)/2) # number of correlations
	@assert(isequal(length(δ), De),
		"mismatch between number of matrix dimension and correlation parameters")

	rLenght = De
	seq = collect(1:De)
	i = ceil.(0.5 .+ 0.5 * sqrt.(1 .+ 8 * seq))  # ith column
	j = seq .- (i .- 2).*(i .- 1)./2             # jth column
	ind1 = (j .- 1) * D .+ i              # linear indexes for lower triangular
	ind2 = (i .- 1) * D .+ j              # linear indexes for upper triangular
	index = convert(Matrix{Int}, hcat(i, j, ind1, ind2, seq))

	if D <= 1
	  error("D must be greater than 1")
	end

	function realtoz(x::Union{T, Vector{T}}, a::Number) where {T <: Number}
	 # DESCRIPTION :
		# Transforms the real line to the interval (-1, 1) using the shifted
		# logistic function

		y = 2 ./ (1 .+ exp.(-a*x)) .- 1
		return y
	end

	function ztoreal(x::Union{T, Vector{T}}, a::Number) where {T <: Number}
	 # DESCRIPTION :
		#  Transforms the interval (-1, 1) to the real line, using the inverse
		# hyperbolic tangent function

		if any(abs.(x) .>= 1)
			error("domain error");
		else
			y = -1/a * log.(2 ./ (x .+ 1) .- 1);
		end
		return y
	end

	function rtoreal(cormap::corrmap, w::Union{T, Vector{T}}) where {T <: Number}
	 	# DESCRIPTION :
	 	# transform the correlation vector in the
		# Transforms the matrix R to the matrix Z (composed by the elements
		# gamma on the real line)

		if length(w) != cormap.De
			error("dimensions do not match")
		end

		D = cormap.D
		ind1 = cormap.index[:, 3];
		ind2 = cormap.index[:, 4];

		R = zeros(D, D);
		R[collect(1:(D+1):(D^2))] = ones(D)
		R[vcat(index[:, 3], index[:, 4])] =	vcat(w, w)
		# display(R)
		Z = zeros(D, D);

		if isposdef(R)
			W = cholesky(R).U
		else
			error("not positive definite")
		end

		Z[1, 2:D] = W[1, 2:D];
		for i1 = 2:(D-1)
			for j1 = (i1+1):D
				Z[i1, j1] = W[i1, j1] .* exp(-0.5*sum(log.(1 .- Z[1:(i1-1), j1].^2)));
			end
		end

		return cormap.ztoreal(Z[ind2], cormap.aval);
	end

	function realtor(cormap::corrmap, w::Union{T, Vector{T}},
		 d::Any) where {T <: Number}
	 	# DESCRIPTION :
		# Transforms the vector on the real line to the matrix R (in a vector);

		if length(w) != cormap.De
			error("dimensions do not match")
		end

		D = cormap.D
		i = cormap.index[:, 1]
		j = cormap.index[:, 2]
		ind2 = cormap.index[:, 4]
		seq = cormap.index[:, end]

		Z = zeros(D, D)
		W = zeros(D, D)
		W[collect(1:(D+1):(D^2))] = ones(D)

		Z[ind2] = cormap.realtoz(w, cormap.aval)

		W[1, 2:D] = Z[1, 2:D]
		for i1 = 2:D
			for j1 = i1:D
				ztmp = 0.5.*sum(log.(1 .- Z[1:(i1-1), j1].^2))
				 W[i1, j1] = ((i1 == j1) + (i1 != j1)*Z[i1, j1]) * exp(ztmp)
			end
		end

		R = transpose(W) * W # correlation matrix
		if isempty(d)
			y = R[ind2]
		elseif d == 1
			y = R
		else
			y = transpose(W)
		end

		return y
	end

	# pass the structure
	corrmap(δ, D, De, index, aval, realtoz, ztoreal, rtoreal, realtor)
end


realrho = eval(Meta.parse(ARGS[1]))
R = length(realrho)
D = Int64((1 + sqrt(1+4*2*R))/2)

# mapping
cmap = corrmap(realrho, D)

# corr Matrix
R = cmap.realtor(cmap, cmap.δ, 1)

print(R)

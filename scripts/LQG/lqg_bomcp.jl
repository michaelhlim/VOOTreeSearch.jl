function BOMCP.x2s(m::LQGPOMDP, x::Vector{Float64})
    s = x
    return s
end

function BOMCP.s2x(m::LQGPOMDP, s::Vector{Float64})
    x = s
    return x
end

function BOMCP.gen_A(m::LQGPOMDP, s::Vector{Float64}, a::Vector{Float64})
    A = zeros(Float64, 3, 3)
    A[1,1] = m.A_matrix[1,1]
    A[1,2] = m.A_matrix[1,2]
    A[2,1] = m.A_matrix[2,1]
    A[2,2] = m.A_matrix[2,2]
    A[3,3] = (s[end] + 1)/(s[end])
    return A
end

function BOMCP.gen_C(m::LQGPOMDP, s::Vector{Float64})
    C = zeros(Float64, 3, 3)
    C[1,1] = m.C_matrix[1,1]
    C[1,2] = m.C_matrix[1,2]
    C[2,1] = m.C_matrix[2,1]
    C[2,2] = m.C_matrix[2,2]
    C[3,3] = 1.0
    return C
end

function BOMCP.vectorize!(v, dims, x::MvNormal)
    v = copy(mean(x))
    return v
end

POMDPs.updater(p::LQGPolicy) = EKFUpdater(p.m, [0.1, 0.1, 1e-300].^2, [0.1, 0.1, 1e-300].^2)

function POMDPs.update(up::EKFUpdater, b::MvNormal, a::Vector{Float64}, o::Vector{Float64})
    μ = mean(b)
    n = length(μ)
    Σ = cov(b)
    s = x2s(up.m, μ)
    # Predict
    sp, z = POMDPs.gen(DDNOut(:sp, :o), up.m, s, a, Random.GLOBAL_RNG)
    xp = s2x(up.m, sp)

    A = gen_A(up.m, μ, a)
    C = gen_C(up.m, xp)

    Σ_hat = A*Σ*transpose(A) + up.Q

    # Update
    y = o - z

    S = C*Σ_hat*transpose(C) + up.R
    K = Σ_hat*transpose(C)/S

    μp = xp + K*y

    Σp = (Matrix{Float64}(I, n, n) - K*C)*Σ_hat

    Σp = round.(Σp, digits=5)
    Σp[end, end] = 1e-300
    bp = MvNormal(μp, Σp)
    return bp
end

function BasicPOMCP.extract_belief(up::EKFUpdater, bn)
    pomcpow_belief = belief(bn).sr_belief

    n_obs = length(pomcpow_belief.dist.cdf)
    X = zeros(Float64, 3, n_obs)
    for (i, item) in enumerate(pomcpow_belief.dist.items)
        X[:,i] = item[1]
    end
    weights = [pomcpow_belief.dist.cdf[1]]
    for i = 2:n_obs
        dw = pomcpow_belief.dist.cdf[i] - pomcpow_belief.dist.cdf[i-1]
        append!(weights, dw)
    end
    W = ProbabilityWeights(weights)

    if n_obs > 1
        μ = mean(X, W, dims=2)[:,1]
        σ = cov(X, W)
        display(σ)
    else
        μ = X[:,1]
        σ = [0.1, 0.1, 1e-300]
        σ = diagm(σ)
        # display(σ)
    end
    # display(μ)
    return MvNormal(μ, σ)
end
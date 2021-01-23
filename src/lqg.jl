##
# defining LQR and LQG problems

struct LQGDists{D1 <: Distribution, D2 <: Distribution, D3 <: Distribution, D4 <: Distribution}
    initial_dist::D1
    V_dist::D2
    W_dist::D3
    W_dist_time::D4
end

struct LQGPOMDP <: POMDP{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    A_matrix::Array{Float64, 2}
    B_matrix::Array{Float64, 2}
    C_matrix::Array{Float64, 2}
    Q_matrix::Array{Float64, 2}
    R_matrix::Array{Float64, 2}
    V_matrix::Array{Float64, 2}
    W_matrix::Array{Float64, 2}
    initial_state::Vector{Float64}
    initial_cov_vector::Vector{Float64}
    dists::LQGDists
    a_space::BoxND
    horizon::Int
end

function LQGPOMDP(;A_matrix::Array{Float64, 2},
                    B_matrix::Array{Float64, 2},
                    C_matrix::Array{Float64, 2},
                    Q_matrix::Array{Float64, 2},
                    R_matrix::Array{Float64, 2},
                    V_matrix::Array{Float64, 2},
                    W_matrix::Array{Float64, 2},
                    initial_state::Vector{Float64},
                    initial_cov_vector::Vector{Float64},
                    a_space::BoxND, horizon::Int)
    
    if norm(initial_cov_vector) == 0
        initial_dist = MvNormal([initial_state; 1.0], 0.0)    
    else
        initial_dist = MvNormal([initial_state; 1.0], diagm([initial_cov_vector; 1e-300]))
    end

    if norm(V_matrix) == 0
        V_dist = MvNormal(size(V_matrix)[1], 0.0)
    else
        V_dist = MvNormal(V_matrix)
    end

    if norm(W_matrix) == 0
        W_dist = MvNormal(size(W_matrix)[1], 0.0)
        W_dist_time = MvNormal(size(W_matrix)[1]+1, 0.0)
    else
        W_dist = MvNormal(W_matrix)
        vzeros = zeros(size(W_matrix)[1], 1)
        hzeros = zeros(1, size(W_matrix)[1] + 1)
        hzeros[1,end] = 1e-300
        W_dist_time = MvNormal(vcat(hcat(W_matrix, vzeros),hzeros))
    end
    
    dists = LQGDists(initial_dist, V_dist, W_dist, W_dist_time)
    return LQGPOMDP(A_matrix, B_matrix, C_matrix, 
                    Q_matrix, R_matrix, V_matrix, 
                    W_matrix, initial_state,
                    initial_cov_vector, dists, a_space, horizon)
end

function update_state(m::LQGPOMDP, s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG)
    x = s[1:end-1]
    t = round(s[end])
    v = rand(rng, m.dists.V_dist)

    xp = m.A_matrix * x + m.B_matrix * a + v
    tp = t + 1
    sp = append!(xp, tp)

    return sp
end

function get_observation(m::LQGPOMDP, s::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG)
    w = rand(rng, m.dists.W_dist)
    obs = m.C_matrix * s[1:end-1] + w
    append!(obs, round(s[end]))
    return obs
end

function get_reward(m::LQGPOMDP, s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64})
    x = s[1:end-1]
    t = round(s[end])

    r = transpose(x) * m.Q_matrix * x
    if t < horizon(m) + 1
        r += transpose(a) * m.R_matrix * a
    end    
    return -1.0 * r
end

function POMDPs.reward(m::LQGPOMDP, s, a, sp)
    get_reward(m, s, a, sp)
end

function POMDPs.gen(m::LQGPOMDP, s::Vector{Float64}, a::Vector{Float64}, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp = update_state(m, s, a, rng=rng)
    o = get_observation(m, sp, rng=rng)
    r = get_reward(m, s, a, sp)
    return (sp=sp, o=o, r=r)
end

function POMDPs.isterminal(m::LQGPOMDP, s::Vector{Float64})
    if round(s[end]) > horizon(m) + 1
        return true
    else
        return false
    end
end

horizon(m::LQGPOMDP) = m.horizon
POMDPs.actions(m::LQGPOMDP) = m.a_space
POMDPs.actiontype(::LQGPOMDP) = Vector{Float64}
POMDPs.discount(::LQGPOMDP) = 1.0 - 1e-16       # avoids overflow issues
POMDPs.initialstate_distribution(m::LQGPOMDP) = m.dists.initial_dist

function POMDPs.observation(m::LQGPOMDP, sp::Vector{Float64})
    xp = sp[1:end-1]
    tp = round(sp[end])
    centered_obs = m.C_matrix * xp
    obs = append!(centered_obs, tp)
    return MvNormal(obs, m.dists.W_dist_time.Σ)
end

max_possible_weight(m::LQGPOMDP, a, o) = 0.0
new_particle(m::LQGPOMDP, a, o) = error("shouldn't get here")

function POMDPModelTools.obs_weight(m::LQGPOMDP, 
                            s::Vector{Float64}, 
                            a::Vector{Float64}, 
                            sp::Vector{Float64}, 
                            o::Vector{Float64})
    xp = sp[1:end-1]
    obs = o[1:end-1]
    centered_obs = obs - m.C_matrix * xp
    return pdf(m.dists.W_dist, centered_obs)
end

function solve_LQG_one_step(m::LQGPOMDP)
    (Q1, Q2) = size(m.Q_matrix)
    timestep = horizon(m)
    S_matrix_array = zeros(Q1, Q2, timestep)
    S_matrix_array[:, :, timestep] = m.Q_matrix

    # Backward pass
    for t in 1:timestep-1
        i = timestep - t
        S_ip1 = S_matrix_array[:, :, i + 1]
        SB = S_ip1 * m.B_matrix
        feedback_gain = inv(transpose(m.B_matrix) * S_ip1 * m.B_matrix + m.R_matrix)
        S_matrix_array[:, :, i] = 
            transpose(m.A_matrix) * (S_ip1  - SB * feedback_gain * transpose(SB)) + m.Q_matrix
    end
    S_ip1 = S_matrix_array[:, :, 1]
    feedback_gain = inv(transpose(m.B_matrix) * S_ip1 * m.B_matrix + m.R_matrix)
    K_matrix = feedback_gain * transpose(m.B_matrix) * S_ip1 * m.A_matrix

    return -1.0 * K_matrix * m.initial_state
end

abstract type LQGPolicy <: Policy end

struct RiccatiLQRPolicy <: LQGPolicy
    m::LQGPOMDP
    P_matrix::Array{Float64, 2}
end

struct ExactLQRPolicy <: LQGPolicy
    m::LQGPOMDP
end
struct RandomLQRPolicy{R <: AbstractRNG} <: LQGPolicy
    m::LQGPOMDP
    rng::R
end

POMDPs.action(p::LQGPolicy, b) = POMDPs.action(p, mean(b))

function POMDPs.action(p::ExactLQRPolicy, s::Vector{Float64}) 
    xbar = s[1:end-1]
    tbar = max(0, s[end])

    (Q1, Q2) = size(p.m.Q_matrix)
    timeleft = horizon(p.m) - convert(Int, tbar) + 1

    if timeleft <= 0
        P_blank = zeros(Q1, Q2)
        return transpose(p.m.B_matrix) * P_blank * p.m.A_matrix * xbar
    end

    P_matrix_array = zeros(Q1, Q2, timeleft)
    P_matrix_array[:, :, timeleft] = p.m.Q_matrix

    # Backward pass
    for t in 1:timeleft-1
        i = timeleft - t
        P_ip1 = P_matrix_array[:, :, i + 1]
        BPA = transpose(p.m.B_matrix) * P_ip1 * p.m.A_matrix
        APA = transpose(p.m.A_matrix) * P_ip1 * p.m.A_matrix
        feedback_gain = inv(transpose(p.m.B_matrix) * P_ip1 * p.m.B_matrix + p.m.R_matrix)
        P_matrix_array[:, :, i] = APA - transpose(BPA) * feedback_gain * BPA + p.m.Q_matrix
    end
    P_ip1 = P_matrix_array[:, :, 1]
    BPA = transpose(p.m.B_matrix) * P_ip1 * p.m.A_matrix
    feedback_gain = inv(transpose(p.m.B_matrix) * P_ip1 * p.m.B_matrix + p.m.R_matrix)
    K_matrix = feedback_gain * BPA

    return -1.0 * K_matrix * xbar
end

function POMDPs.action(p::RiccatiLQRPolicy, s::Vector{Float64}) 
    xbar = s[1:end-1]
    tbar = max(0, s[end])

    feedback_gain = inv(p.m.R_matrix + transpose(p.m.B_matrix) * p.P_matrix * p.m.B_matrix)
    BPA = transpose(p.m.B_matrix) * p.P_matrix * p.m.A_matrix
    F_matrix = feedback_gain * BPA

    return -1.0 * F_matrix * xbar
end

POMDPs.action(p::RandomLQRPolicy, s::Vector{Float64}) = rand(p.rng, p.m.a_space)
    
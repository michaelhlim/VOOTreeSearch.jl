# using Revise
using ClusterManagers
using Distributed

worker_ids = if "SLURM_JOBID" in keys(ENV)
    wids = ClusterManagers.addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    wids
else
    addprocs(2)
end  

Distributed.@everywhere begin
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
end

Distributed.@everywhere begin
    using POMDPs
    using BasicPOMCP
    using POMCPOW
    using VOOTreeSearch
    using ParticleFilters
    using CPUTime
    using DataFrames
    using ARDESPOT
    using Distributions
    using VDPTag2
    using ProgressMeter
    using Dates
    using CSV
    using POMDPSimulators
    using Random
    using Statistics
    using StatsBase
    using LinearAlgebra
    using VDPTag2Experiments
    using LinearAlgebra: diagm, PosDefException, det, eigvals, eigvecs
    using Random: MersenneTwister, seed!
    using Distances: euclidean
    import MatrixEquations: ared

    using BOMCP
    include("./scripts/LQG/lqg_bomcp.jl")
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

@show max_time = 1.0
@show max_depth = 5
@show max_query = 100

@show start_mean = [65.0, 30.0, 2.5, 30.0, 3.5, 20.0, 0.5]
@show start_cov = diagm([100.0^2, 10.0^2, 2.0^2, 10.0^2, 2.0^2, 10.0^2, 0.5^2])
d = MvNormal(start_mean, start_cov)
on_cluster = true
save_data = true
parallelize = true

if on_cluster
  K = 150 # number of parameter samples
  n = 20 # number of evaluation simulations
  m = 30 # number of elite samples
  max_iters = 100
else
  K = 15 # number of parameter samples
  n = 1 # number of evaluation simulations
  m = 10 # number of elite samples
  max_iters = 3 
end


# LQG
N = 2
R_gain = 1
V_sig = 0.1
W_sig = 0.1
init_cov_sig = 0.1

eye = [1.0 0.0 ; 0.0 1.0]
A_mat = eye
B_mat = eye
C_mat = eye
Q_mat = eye
R_mat = R_gain * eye
V_mat = V_sig^2 * eye
W_mat = W_sig^2 * eye
init_state_time = [-10.0 ; 10.0 ; 1.0]
init_state = init_state_time[1:end-1]
init_cov = init_cov_sig^2 * [1.0 ; 1.0]
a_space = BoxND(2, [-10.0, -10.0], [10.0, 10.0])

@show pomdp = LQGPOMDP(A_matrix = A_mat,
            B_matrix = B_mat,
            C_matrix = C_mat,
            Q_matrix = Q_mat,
            R_matrix = R_mat,
            V_matrix = V_mat,
            W_matrix = W_mat,
            initial_state = init_state,
            initial_cov_vector = init_cov,
            a_space = a_space,
            horizon = N)

DARESolution, _, _ = ared(A_mat, B_mat, R_mat, Q_mat);

Distributed.@everywhere begin        
    include("./scripts/LQG/lqg_bomcp.jl")

    struct SimWrap
        x::Vector{Float64}
        k::Int64
        i::Int64
        max_time::Float64
        max_depth::Int64
        max_query::Int64
        pomdp::POMDP
        DARESolution::Any
    end

    function gen_sim(sim_wrap::SimWrap)
        x = sim_wrap.x
        k = sim_wrap.k
        i = sim_wrap.i
        max_time = sim_wrap.max_time
        max_depth = sim_wrap.max_depth
        max_query = sim_wrap.max_query
        pomdp = sim_wrap.pomdp
        DARESolution = sim_wrap.DARESolution

        c = x[1]
        @assert c >= 0.0
        k_act = x[2]
        @assert k_act >= 1.0
        inv_alpha_act = x[3]
        @assert inv_alpha_act >= 0.1
        k_obs = x[4]
        @assert k_obs >= 1.0
        inv_alpha_obs = x[5]
        @assert inv_alpha_obs >= 0.1
        log_kernel = x[6]
        @assert log_kernel >= 1.0
        lambda = x[7]
        @assert lambda >= 0.1

        rng = MersenneTwister(7)
        belief_updater = EKFUpdater(pomdp, [0.1, 0.1, 1e-300].^2, [0.1, 0.1, 1e-300].^2)
        rollout_policy = RiccatiLQRPolicy(pomdp, DARESolution)

        action_selector = BOActionSelector(2, # action dims
                                    3, #belief dims
                                    false, #discrete actions
                                    kernel_params=[log(log_kernel), 0.0],
                                    k_neighbors = 5,
                                    belief_λ = lambda,
                                    lower_bounds = [-10.0, -10.0],
                                    upper_bounds = [10.0, 10.0],
                                    buffer_size=100,
                                    initial_action=rollout_policy
                                    )

        solver = BOMCPSolver(action_selector, belief_updater,
                    depth=max_depth, n_iterations=max_query,
                    max_time = max_time,
                    exploration_constant=c,
                    k_belief = k_obs,
                    alpha_belief = 1/inv_alpha_obs,
                    k_action = k_act,
                    alpha_action = 1/inv_alpha_act,
                    estimate_value=BOMCP.PORollout(rollout_policy, belief_updater),
                    rng=rng
                    )

        planner = solve(solver, pomdp)
        filter = EKFUpdater(pomdp, [0.1, 0.1, 1e-300].^2, [0.1, 0.1, 1e-300].^2)
        
        seed!(planner, i+40000*k)
        sim = Sim(pomdp,
                  planner,
                  filter,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=50,
                  metadata=Dict(:i=>i, :k=>k)
                 )

        result = POMDPs.simulate(sim)
        output = (reward=discounted_reward(result),)
        return merge(sim.metadata, output)
    end
end   

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
results_csv = DataFrame(c=Float64[], k_act=Float64[], inv_alpha_act=Float64[], k_obs=Float64[], inv_alpha_obs=Float64[], log_kernel=Float64[], lambda=Float64[])
try
    for i in 1:max_iters
        sims = SimWrap[]
        params = Vector{Float64}[]
        println(stderr,"creating $K simulation sets")
        for k in 1:K
            p = rand(d)
            p[1] = max(0.0, p[1])
            p[2] = max(1.0, p[2])
            p[3] = max(0.1, p[3])
            p[4] = max(1.0, p[4])
            p[5] = max(0.1, p[5])
            p[6] = max(1.0, p[6])
            p[7] = max(0.1, p[7])
            push!(params, p)
            for j in 1:n
                sim_wrap = SimWrap(p, k, j, max_time, max_depth, max_query, deepcopy(pomdp), DARESolution)
                push!(sims, sim_wrap)
            end
        end
        @assert length(params) == K
        if parallelize
            results = @showprogress pmap(gen_sim, sims)
        else
            results = @showprogress map(gen_sim, sims)
        end
        results = POMDPSimulators.create_dataframe(results)
        combined = by(results, :k) do df
            DataFrame(mean_reward=mean(df[:reward]))
        end
        @show mean(combined[:mean_reward])
        order = sortperm(combined[:mean_reward])
        elite = params[combined[:k][order[K-m:end]]]
        elite_matrix = Matrix{Float64}(undef, length(start_mean), m)
        for k in 1:m
            elite_matrix[:,k] = elite[k]
        end
        try
            global d = fit(typeof(d), elite_matrix)
        catch ex
            if ex isa PosDefException
                println(stderr,"pos def exception")
                global d = fit(typeof(d), elite_matrix += 0.01*randn(size(elite_matrix)))
            else
                rethrow(ex)
            end
        end
        println(stderr, "Iteration $i")
        println(stderr, "Mean: ", mean(d))
        println(stderr, "Cov: ", det(cov(d)))
        ev = eigvals(cov(d))
        println(stderr, "Cov (eig): ", ev)
        for j in 1:length(ev)
            println(stderr, "Eigvecs: ", eigvecs(cov(d))[:,j])
        end
        push!(results_csv, mean(d))
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "LQG_bomcp_opt"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end
finally
    Distributed.rmprocs(worker_ids)
end

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

    using BOMCP
    using Statistics
    using StatsBase
    include("./deps/BOMCP.jl/scripts/lander_po.jl")

    function BOMCP.vectorize!(v, dims, x::MvNormal)
        v = copy(mean(x))
        return v
    end
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

max_time = 1.0
max_depth = 250

start_mean = [20.0, 7.5, 4.0, 2.0, 6.0, 15.0, 0.5]
start_cov = diagm([10.0^2, 5.0^2, 5.0^2, 5.0^2, 5.0^2, 10.0^2, 0.5^2])
d = MvNormal(start_mean, start_cov)
on_cluster = true
save_data = true
parallelize = true

if on_cluster
  max_query = 10_000
  K = 150 # number of parameter samples
  n = 20 # number of evaluation simulations
  m = 30 # number of elite samples
  max_iters = 50
else
  max_query = 100
  K = 15 # number of parameter samples
  n = 1 # number of evaluation simulations
  m = 10 # number of elite samples
  max_iters = 3 
end


# Lander
pomdp = LunarLander()

println(stderr, "Max time: ", max_time)
println(stderr, "Max depth: ", max_depth)
println(stderr, "Max query: ", max_query)
println(stderr, "Start mean: ", start_mean)
println(stderr, "Start cov: ", start_cov)
println(stderr, "POMDP: ", pomdp)

Distributed.@everywhere begin        
    struct SimWrap
        x::Vector{Float64}
        k::Int64
        i::Int64
        max_time::Float64
        max_depth::Int64
        max_query::Int64
        pomdp::POMDP
    end

    function gen_sim(sim_wrap::SimWrap)
        x = sim_wrap.x
        k = sim_wrap.k
        i = sim_wrap.i
        max_time = sim_wrap.max_time
        max_depth = sim_wrap.max_depth
        max_query = sim_wrap.max_query
        pomdp = sim_wrap.pomdp

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
        belief_updater = EKFUpdater(pomdp, pomdp.Q.^2, pomdp.R.^2)
        rollout_policy = LanderPolicy(pomdp)

        action_selector = BOActionSelector(3, # action dims
                                    6, #belief dims
                                    false, #discrete actions
                                    kernel_params=[log(log_kernel), 0.0],
                                    k_neighbors = 5,
                                    belief_λ = lambda,
                                    lower_bounds = [-5.0, 0.0, -1.0],
                                    upper_bounds = [5.0, 15.0, 1.0],
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
                    estimate_value=BOMCP.RolloutEstimator(rollout_policy),
                    default_action=BOMCP.ReportWhenUsed([0.0, 0.0, 0.0]),
                    rng=rng
                    )

        planner = solve(solver, pomdp)
        
        seed!(planner, i+40000*k)
        sim = Sim(pomdp,
                  planner,
                  belief_updater,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=200,
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
                sim_wrap = SimWrap(p, k, j, max_time, max_depth, max_query, deepcopy(pomdp))
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
        println(stderr, "Reward (mean): ", mean(combined[:mean_reward]))
        println(stderr, "Mean: ", mean(d))
        println(stderr, "Cov: ", det(cov(d)))
        ev = eigvals(cov(d))
        println(stderr, "Cov (eig): ", ev)
        for j in 1:length(ev)
            println(stderr, "Eigvecs: ", eigvecs(cov(d))[:,j])
        end
        push!(results_csv, mean(d))
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "Lander_bomcp_opt"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end
finally
    Distributed.rmprocs(worker_ids)
end

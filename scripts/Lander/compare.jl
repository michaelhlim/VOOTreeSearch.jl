# using Revise
using ClusterManagers
using Distributed

worker_ids = if "SLURM_JOBID" in keys(ENV)
    wids = ClusterManagers.addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    wids
else
    addprocs(4)
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
    using Printf
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

max_depth = 250
solver_list = Dict("POMCPOW"=>[10.0, 3.0, 4.0, 2.0, 10.0], 
                    "VOMCPOW"=>[30.0, 4.0, 4.0, 1.5, 5.0, 0.9], 
                    "BOMCP"=>[10.0, 3.0, 4.0, 2.0, 10.0, 15.0, 0.5],)
on_cluster = true
save_data = true
parallelize = true

if on_cluster
    max_time_list = 10.0 .^ (-2:0.25:0)
    max_query = 100_000
    n_iters = 1000
else
    max_time_list = 10.0 .^ (-2:0.5:0)
    max_query = 100_000
    n_iters = 10 
end

# Lander
pomdp = LunarLander(dt=0.4)

println(stderr, "Max times: ", max_time_list)
println(stderr, "Max depth: ", max_depth)
println(stderr, "Max query: ", max_query)
println(stderr, "Number of iterations: ", n_iters)
println(stderr, "Solvers: ", solver_list)
println(stderr, "POMDP: ", pomdp)

Distributed.@everywhere begin        
    struct SimWrap
        x::Vector{Float64}
        solver_name::String
        i::Int64
        max_time::Float64
        max_depth::Int64
        max_query::Int64
        pomdp::POMDP
    end

    function gen_sim(sim_wrap::SimWrap)
        x = sim_wrap.x
        solver_name = sim_wrap.solver_name
        i = sim_wrap.i
        max_time = sim_wrap.max_time
        max_depth = sim_wrap.max_depth
        max_query = sim_wrap.max_query
        pomdp = sim_wrap.pomdp

        rng = MersenneTwister(7)
        belief_updater = EKFUpdater(pomdp, pomdp.Q.^2, pomdp.R.^2)
        rollout_policy = LanderPolicy(pomdp)

        c = x[1]
        k_act = x[2]
        inv_alpha_act = x[3]
        k_obs = x[4]
        inv_alpha_obs = x[5]

        if solver_name == "POMCPOW"
            solver = POMCPOWSolver(tree_queries=max_query,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=POMCPOW.RolloutEstimator(rollout_policy),
                                   next_action=RolloutFirst(rng, rollout_policy),
                                   default_action=BasicPOMCP.ReportWhenUsed([0.0, 0.0, 0.0]),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            
            seed!(planner, i+40000)
            sim = Sim(deepcopy(pomdp),
                      planner,
                      belief_updater,
                      rng=MersenneTwister(i+50_000),
                      max_steps=200,
                      metadata=Dict(:i=>i, :solver=>solver_name)
                     )

        elseif solver_name == "VOMCPOW"
            p_voo = x[6]
            sigs = [0.2, 0.5, 0.05]
            sig = diagm(sigs)
            epsilon = 0.01
            voo_opt = VOOActionGenerator(p_voo, Distributions.Uniform(), MvNormal(sig), rng, epsilon, true, 20)

            solver = POMCPOWSolver(tree_queries=max_query,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=POMCPOW.RolloutEstimator(rollout_policy),
                                   next_action = VOORolloutFirst(voo_opt, rollout_policy),
                                   default_action=BasicPOMCP.ReportWhenUsed([0.0, 0.0, 0.0]),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            
            seed!(planner, i+40000)
            sim = Sim(deepcopy(pomdp),
                      planner,
                      belief_updater,
                      rng=MersenneTwister(i+50_000),
                      max_steps=200,
                      metadata=Dict(:i=>i, :solver=>solver_name)
                     )

        elseif solver_name == "BOMCP"
            max_query = Int64(max_query/10)
            log_kernel = x[6]
            lambda = x[7]

            belief_updater = EKFUpdater(pomdp, pomdp.Q.^2, pomdp.R.^2)

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
            
            seed!(planner, i+40000)
            sim = Sim(pomdp,
                      planner,
                      belief_updater,
                      rng=MersenneTwister(i+50_000),
                      max_steps=200,
                      metadata=Dict(:i=>i, :solver=>solver_name)
                     )
        end

        result = POMDPs.simulate(sim)
        output = (reward=discounted_reward(result),)
        return merge(sim.metadata, output)
    end
end   

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
results_csv = DataFrame(max_time=Float64[], solver=String[], mean_reward=Float64[], se_reward=Float64[])

try
    for iter_time in max_time_list
        sims = SimWrap[]
        for solver_name in keys(solver_list)
            for j in 1:n_iters
                sim_wrap = SimWrap(solver_list[solver_name], solver_name, j, iter_time, max_depth, max_query, deepcopy(pomdp))
                push!(sims, sim_wrap)
            end
        end
        if parallelize
            results = @showprogress pmap(gen_sim, sims)
        else
            results = @showprogress map(gen_sim, sims)
        end
        println(stderr, "Max time: $iter_time :")
        results = POMDPSimulators.create_dataframe(results)
        combined = by(results, :solver) do df
            DataFrame(mean_reward=mean(df[:reward]), se_reward=(std(df[:reward])/sqrt(n_iters)))
        end
        for row in eachrow(combined)
            result_row = [iter_time, row.solver, row.mean_reward, row.se_reward]
            println(stderr, @sprintf("- %s reward: %6.3f ± %6.3f", row.solver, row.mean_reward, row.se_reward))
            push!(results_csv, result_row)  
        end
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "Lander_compare_results"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end
finally
    Distributed.rmprocs(worker_ids)
end

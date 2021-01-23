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

    using MCTS
    using StaticArrays
    using POMDPModelTools: GenerativeBeliefMDP
    include("./scripts/VDPTag2/vdp_next_action.jl")
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

max_depth = 10
solver_list = Dict("POMCPOW"=>[110.0, 30.0, 30.0, 5.0, 100.0], 
                    "VOMCPOW"=>[85.0, 30.0, 30.0, 2.5, 100.0, 0.7])

on_cluster = true
save_data = true
parallelize = true

if on_cluster
    max_time_list = 10.0 .^ (-2:0.25:0)
    max_query = 100_000
    n_iters = 1000
else
    # max_time_list = 10.0 .^ (-2:0.5:0)
    max_time_list = [0.1]
    max_query = 100_000
    n_iters = 50 
end

# VDPTag2
pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))

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

        c = x[1]
        k_act = x[2]
        inv_alpha_act = x[3]
        k_obs = x[4]
        inv_alpha_obs = x[5]
        
        rng = MersenneTwister(7)
        rollout_policy = ToNextMLSolver(rng)

        if solver_name == "POMCPOW"
            solver = POMCPOWSolver(tree_queries=max_query,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=FORollout(rollout_policy),
                                   next_action=RootToNextMLFirst(rng),
                                   default_action=BasicPOMCP.ReportWhenUsed(TagAction(false, 0.0)),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            filter = SIRParticleFilter(deepcopy(pomdp), 
                                       100_000, 
                                       rng=MersenneTwister(i+90_000))
            
            seed!(planner, i+40000)
            sim = Sim(deepcopy(pomdp),
                      planner,
                      filter,
                      rng=MersenneTwister(i+50_000),
                      max_steps=100,
                      metadata=Dict(:i=>i, :solver=>solver_name)
                     )

        elseif solver_name == "VOMCPOW"
            p_voo = x[6]
            sig = diagm([0.1])
            epsilon = 0.1/10.0
            voo_opt = VOOActionGenerator(p_voo, Distributions.Uniform(), MvNormal(sig), rng, epsilon, true, 20)

            solver = POMCPOWSolver(tree_queries=max_query,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=FORollout(rollout_policy),
                                   next_action = RootToNextMLFirstVOO(voo_opt),
                                   default_action=BasicPOMCP.ReportWhenUsed(TagAction(false, 0.0)),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            filter = SIRParticleFilter(deepcopy(pomdp), 
                                       100_000, 
                                       rng=MersenneTwister(i+90_000))
            
            seed!(planner, i+40000)
            sim = Sim(deepcopy(pomdp),
                      planner,
                      filter,
                      rng=MersenneTwister(i+50_000),
                      max_steps=100,
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
        println(stderr, results)
        combined = by(results, :solver) do df
            DataFrame(mean_reward=mean(df[:reward]), se_reward=(std(df[:reward])/sqrt(n_iters)))
        end
        for row in eachrow(combined)
            result_row = [iter_time, row.solver, row.mean_reward, row.se_reward]
            println(stderr, @sprintf("- %s reward: %6.3f ± %6.3f", row.solver, row.mean_reward, row.se_reward))
            push!(results_csv, result_row)  
        end
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "VDP_compare_results"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end
finally
    Distributed.rmprocs(worker_ids)
end

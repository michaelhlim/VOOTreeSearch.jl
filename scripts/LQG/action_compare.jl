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

    # # Requires BOMCP
    # using BOMCP
    using Statistics
    using StatsBase
    import MatrixEquations: ared
    # include("./scripts/LQG/lqg_bomcp.jl")

    # function BOMCP.vectorize!(v, dims, x::MvNormal)
    #     v = copy(mean(x))
    #     return v
    # end
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

solver_list = Dict("POMCPOW"=>[65.0, 30.0, 2.5, 30.0, 4.0], 
                    "VOMCPOW"=>[60.0, 25.0, 5.5, 25.0, 2.5, 0.8],
                    # "BOMCP"=>[135.0, 30.0, 4.0, 20.0, 4.0, 15.0, 0.4], # Requires BOMCP
                    )
on_cluster = true
save_data = true
parallelize = true
use_lqr_policy = true
use_riccati_policy = true
use_random_policy = false

if on_cluster
    max_depth = 3
    max_time = 1.0
    n_iters = 1000
    push!(solver_list, ("VOWSS"=>[10, 200, 0.8, 0.4, 2]))
else
    max_depth = 3
    max_time = 1.0
    n_iters = 20
    push!(solver_list, ("VOWSS"=>[5, 10, 0.8, 0.4, 2]))
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
initial_belief = MvNormal([init_state; 1.0], diagm([init_cov; 1e-300]))


pomdp = LQGPOMDP(A_matrix = A_mat,
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

println(stderr, "Max time: ", max_time)
println(stderr, "Max depth: ", max_depth)
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
        pomdp::POMDP
        initial_belief::Any
        lqg_solution::Vector{Float64}
        use_lqr_policy::Bool
        use_riccati_policy::Bool
        DARE::Array{Float64,2}
    end

    function gen_sim(sim_wrap::SimWrap)
        x = sim_wrap.x
        solver_name = sim_wrap.solver_name
        i = sim_wrap.i
        max_time = sim_wrap.max_time
        max_depth = sim_wrap.max_depth
        pomdp = sim_wrap.pomdp
        use_lqr_policy = sim_wrap.use_lqr_policy
        use_riccati_policy = sim_wrap.use_riccati_policy
        DARE = sim_wrap.DARE

        c = x[1]
        k_act = x[2]
        inv_alpha_act = x[3]
        k_obs = x[4]
        inv_alpha_obs = x[5]
        
        rng = MersenneTwister(7)
        if use_lqr_policy
            rollout_policy = ExactLQRPolicy(pomdp)
        elseif use_riccati_policy
            rollout_policy = RiccatiLQRPolicy(pomdp, DARE)
        else
            rollout_policy = RandomLQRPolicy(pomdp, rng)
        end

        if solver_name == "POMCPOW"
            solver = POMCPOWSolver(tree_queries=1_000,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=FORollout(rollout_policy),
                                   next_action=RolloutFirst(rng, rollout_policy),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            seed!(planner, i+40000)

        elseif solver_name == "VOMCPOW"
            p_voo = x[6]
            sig = diagm([0.5, 0.5])
            epsilon = 0.5/10.0
            voo_opt = VOOActionGenerator(p_voo, Distributions.Uniform(), MvNormal(sig), rng, epsilon, true, 20)

            solver = POMCPOWSolver(tree_queries=1_000,
                                   criterion=MaxUCB(c),
                                   max_depth=max_depth,
                                   max_time=max_time,
                                   k_action=k_act,
                                   alpha_action=1/inv_alpha_act,
                                   k_observation=k_obs,
                                   alpha_observation=1/inv_alpha_obs,
                                   estimate_value=FORollout(rollout_policy),
                                   next_action=VOORolloutFirst(voo_opt, rollout_policy),
                                   rng=rng
                                  )

            planner = solve(solver, pomdp)
            seed!(planner, i+40000)

        # # Requires BOMCP
        # elseif solver_name == "BOMCP"
        #     log_kernel = x[6]
        #     lambda = x[7]
        #     belief_updater = EKFUpdater(pomdp, [0.1, 0.1, 1e-300].^2, [0.1, 0.1, 1e-300].^2)

        #     action_selector = BOActionSelector(2, # action dims
        #                                 3, #belief dims
        #                                 false, #discrete actions
        #                                 kernel_params=[log(log_kernel), 0.0],
        #                                 k_neighbors = 5,
        #                                 belief_λ = lambda,
        #                                 lower_bounds = [-10.0, -10.0],
        #                                 upper_bounds = [10.0, 10.0],
        #                                 buffer_size=100,
        #                                 initial_action=rollout_policy
        #                                 )

        #     solver = BOMCPSolver(action_selector, belief_updater,
        #                 depth=max_depth, n_iterations=100,
        #                 max_time = max_time,
        #                 exploration_constant=c,
        #                 k_belief = k_obs,
        #                 alpha_belief = 1/inv_alpha_obs,
        #                 k_action = k_act,
        #                 alpha_action = 1/inv_alpha_act,
        #                 estimate_value=BOMCP.RolloutEstimator(rollout_policy),
        #                 rng=rng
        #                 )

        #     planner = solve(solver, pomdp)
        #     seed!(planner, i+40000)
        
        elseif solver_name == "VOWSS"
            state_width = x[1]
            action_width = x[2]
            vowss_p = x[3]
            action_width_decay = x[4]
            cont_action_dim = x[5]
            
            vowss_sig = diagm([0.5, 0.5])
            vowss_epsilon = 0.5/10.0
            voo = VOOActionGenerator(vowss_p, Distributions.Uniform(), MvNormal(vowss_sig), rng, vowss_epsilon)
            vowss_opt = VOWSSOptions(max_depth, state_width, action_width, action_width_decay, true, voo)

            solver = VOOSparseSamplingSolver(vowss_opt, rng)
            planner = solve(solver, deepcopy(pomdp))
            seed!(planner.rng, i+40000)
        end

        best_action = POMDPs.action(planner, sim_wrap.initial_belief)
        output = (solver = solver_name, action_x = best_action[1], action_y = best_action[2], dist = euclidean(best_action, sim_wrap.lqg_solution))
        return output
    end
end   

DARESolution, _, _ = ared(A_mat, B_mat, R_mat, Q_mat);
lqg_solution = solve_LQG_one_step(pomdp)
datestring = Dates.format(now(), "e_d_u_Y_HH_MM")

try
    ########## LQR Policy
    if use_lqr_policy
        sims = SimWrap[]
        results_csv = DataFrame(solver=String[], action_x=Float64[], action_y=Float64[], dist=Float64[])
        for solver_name in keys(solver_list)
            for j in 1:n_iters
                sim_wrap = SimWrap(solver_list[solver_name], solver_name, j, max_time, max_depth, deepcopy(pomdp), deepcopy(initial_belief), lqg_solution, true, false, DARESolution)
                push!(sims, sim_wrap)
            end
        end
        println(stderr, "Starting simulations...")
        if parallelize
            results = @showprogress pmap(gen_sim, sims)
        else
            results = @showprogress map(gen_sim, sims)
        end
        results = POMDPSimulators.create_dataframe(results)
        for row in eachrow(results)
            result_row = [row.solver, row.action_x, row.action_y, row.dist]
            push!(results_csv, result_row)  
        end
        combined = by(results, :solver) do df
            DataFrame(mean_x=mean(df[:action_x]), mean_y=mean(df[:action_y]), mean_dist=mean(df[:dist]), se_dist=(std(df[:dist])/sqrt(n_iters)))
        end
        for row in eachrow(combined)
            println(stderr, @sprintf("- %s mean action: [%6.3f, %6.3f], dist to LQG: %6.3f ± %6.3f", row.solver, row.mean_x, row.mean_y, row.mean_dist, row.se_dist))
        end
        
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "LQG_action_compare_LQR_results"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end

    ########## Saturation Policy
    if use_riccati_policy
        sims = SimWrap[]
        results_csv = DataFrame(solver=String[], action_x=Float64[], action_y=Float64[], dist=Float64[])
        for solver_name in keys(solver_list)
            if use_lqr_policy && solver_name == "VOWSS"
                continue # Don't simulate VOWSS again if doing both types of simulation
            end
            for j in 1:n_iters
                sim_wrap = SimWrap(solver_list[solver_name], solver_name, j, max_time, max_depth, deepcopy(pomdp), deepcopy(initial_belief), lqg_solution, false, true, DARESolution)
                push!(sims, sim_wrap)
            end
        end
        println(stderr, "Starting simulations...")
        if parallelize
            results = @showprogress pmap(gen_sim, sims)
        else
            results = @showprogress map(gen_sim, sims)
        end
        results = POMDPSimulators.create_dataframe(results)
        for row in eachrow(results)
            result_row = [row.solver, row.action_x, row.action_y, row.dist]
            push!(results_csv, result_row)  
        end
        combined = by(results, :solver) do df
            DataFrame(mean_x=mean(df[:action_x]), mean_y=mean(df[:action_y]), mean_dist=mean(df[:dist]), se_dist=(std(df[:dist])/sqrt(n_iters)))
        end
        for row in eachrow(combined)
            println(stderr, @sprintf("- %s mean action: [%6.3f, %6.3f], dist to LQG: %6.3f ± %6.3f", row.solver, row.mean_x, row.mean_y, row.mean_dist, row.se_dist))
        end
        
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "LQG_action_compare_Riccati_results"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end

    ########## Random Policy
    if use_random_policy
        sims = SimWrap[]
        results_csv = DataFrame(solver=String[], action_x=Float64[], action_y=Float64[], dist=Float64[])
        for solver_name in keys(solver_list)
            if solver_name == "VOWSS"
                continue # Don't simulate VOWSS again if doing both types of simulation
            end
            for j in 1:n_iters
                sim_wrap = SimWrap(solver_list[solver_name], solver_name, j, max_time, max_depth, deepcopy(pomdp), deepcopy(initial_belief), lqg_solution, false, false, DARESolution)
                push!(sims, sim_wrap)
            end
        end
        println(stderr, "Starting simulations...")
        if parallelize
            results = @showprogress pmap(gen_sim, sims)
        else
            results = @showprogress map(gen_sim, sims)
        end
        results = POMDPSimulators.create_dataframe(results)
        for row in eachrow(results)
            result_row = [row.solver, row.action_x, row.action_y, row.dist]
            push!(results_csv, result_row)  
        end
        combined = by(results, :solver) do df
            DataFrame(mean_x=mean(df[:action_x]), mean_y=mean(df[:action_y]), mean_dist=mean(df[:dist]), se_dist=(std(df[:dist])/sqrt(n_iters)))
        end
        for row in eachrow(combined)
            println(stderr, @sprintf("- %s mean action: [%6.3f, %6.3f], dist to LQG: %6.3f ± %6.3f", row.solver, row.mean_x, row.mean_y, row.mean_dist, row.se_dist))
        end
        
        if save_data
            fname_csv = joinpath(@__DIR__, "../data", "LQG_action_compare_Random_results"*"_"*datestring*".csv")
            CSV.write(fname_csv, results_csv)
            println(stderr,"Saving parameters to ", fname_csv)
        end
    end
finally
    Distributed.rmprocs(worker_ids)
end

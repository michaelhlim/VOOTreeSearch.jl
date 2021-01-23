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
	using POMDPModels
	using POMDPSimulators
	using POMDPPolicies
	using POMDPModelTools
	using BeliefUpdaters
	using POMCPOW
	using BasicPOMCP
	using VOOTreeSearch
	using ParticleFilters
	using Distributions
	using Random
	using VDPTag2
	using ProgressMeter
	using DataFrames
	using Statistics
	using MCTS
	using LinearAlgebra
	using Dates
	using CSV
	import Distances: euclidean
end

on_cluster = true
save_data = true

if on_cluster
    n_iter = 1000
    state_width_list = [1, 3, 5, 7, 10]
	action_width_list = [50, 100, 150, 200]
else
    n_iter = 10
    state_width_list = [1, 2]
	action_width_list = [10, 20]
end

# VOWSS parameters
depth = 3
vowss_p = 0.8
vowss_sig = 0.5
action_width_decay = 0.4
vowss_epsilon = vowss_sig/10.0

# LQG test
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
init_state_time = [-10.0 ; 10.0 ; 0.0]
init_state = init_state_time[1:end-1]
init_cov = init_cov_sig^2 * [1.0 ; 1.0]
a_space = BoxND(2, [-10.0, -10.0], [10.0, 10.0])
belief = MvNormal([init_state; 1.0], diagm([init_cov; 1e-16]))


lqg_pomdp = LQGPOMDP(A_matrix = A_mat,
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

@everywhere begin
	abstract type Sims end

	struct MySims <: Sims
	    pomdp::POMDP
	    solver::Any
	    initial_belief::Any
	    lqg_solution::Any
	end

    function get_action_dist(sim::Sims)
    	solver = solve(sim.solver, sim.pomdp)
		a = action(solver, sim.initial_belief)
		return euclidean(a, sim.lqg_solution)
    end

    function gen_sims(pomdp::POMDP, state_width, action_width, lqg_solution, n_iter)
		sims = []
		for i = 1:n_iter
			cont_action_dim = 2
			rng = MersenneTwister(10_000 + i)

			voo = VOOActionGenerator(vowss_p, Distributions.Uniform(), MvNormal(cont_action_dim, vowss_sig), 
				rng, vowss_epsilon)
			vowss_opt = VOWSSOptions(depth, state_width, action_width, action_width_decay, true, voo)
			solver = VOOSparseSamplingSolver(vowss_opt, rng)

			sim = MySims(deepcopy(pomdp), solver, belief, deepcopy(lqg_solution))
			push!(sims, sim)
		end
		return sims
	end
end
		    
lqg_solution = solve_LQG_one_step(lqg_pomdp)
results_csv = DataFrame(s_width=Float64[], a_width=Float64[], dist_mean=Float64[], dist_se=Float64[])
datestring = Dates.format(now(), "e_d_u_Y_HH_MM")

try
	println(stderr, "VOWSS Results:")
	for state_width in state_width_list
		for action_width in action_width_list
			sims = gen_sims(lqg_pomdp, state_width, action_width, lqg_solution, n_iter)
			result = @showprogress pmap(get_action_dist, sims)
			println(stderr, "State Width: ", state_width, ", Action Width: ", action_width, " -- mean dist: ", mean(result), ", se dist: ", (std(result)/sqrt(n_iter)))
			push!(results_csv, (state_width, action_width, mean(result), (std(result)/sqrt(n_iter))))
		end
	end

	println(stderr, results_csv)
	###### SAVE DATA ######

	if save_data
        fname_csv = joinpath(@__DIR__, "../data", "LQG_vowss_action_table"*"_"*datestring*".csv")
        CSV.write(fname_csv, results_csv)
        println(stderr,"Saving parameters to ", fname_csv)
    end
finally
	Distributed.rmprocs(worker_ids)
end
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
  using Dates
  using CSV
  using POMDPSimulators
  using VDPTag2Experiments
  using LinearAlgebra: diagm, PosDefException, det, eigvals, eigvecs
  using Random: MersenneTwister, seed!
  using Distances: euclidean
  import MatrixEquations: ared
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

@show max_time = 1.0
@show max_depth = 5
@show max_query = 1_000

@show start_mean = [65.0, 30.0, 2.5, 30.0, 4.0]
@show start_cov = diagm([100.0^2, 10.0^2, 2.0^2, 10.0^2, 2.0^2])
d = MvNormal(start_mean, start_cov)
on_cluster = true
save_data = true
parallelize = true

if on_cluster
  K = 150 # number of parameter samples
  n = 20 # number of evaluation simulations
  m = 30 # number of elite samples
  max_iters = 50
else
  K = 10 # number of parameter samples
  n = 1 # number of evaluation simulations
  m = 7 # number of elite samples
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

function gen_sims(x::Vector{Float64}, n, k, seed)
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

    sims = []

    for i in 1:n
        rng = MersenneTwister(7)

        ro = RiccatiLQRPolicy(pomdp, DARESolution)
        
        solver = POMCPOWSolver(tree_queries=max_query,
                               criterion=MaxUCB(c),
                               max_depth=max_depth,
                               max_time=max_time,
                               k_action=k_act,
                               alpha_action=1/inv_alpha_act,
                               k_observation=k_obs,
                               alpha_observation=1/inv_alpha_obs,
                               estimate_value=FORollout(ro),
                               next_action=RolloutFirst(rng, ro),
                               rng=rng
                              )

        planner = solve(solver, pomdp)
        filter = SIRParticleFilter(deepcopy(pomdp), 
                                   max_query*10, 
                                   rng=MersenneTwister(i+90_000))
        
        seed!(planner, i+40000*k)
        sim = Sim(deepcopy(pomdp),
                  planner,
                  filter,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=50,
                  metadata=Dict(:i=>i, :k=>k)
                 )
        push!(sims, sim)
    end
    
    return sims
end

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")
results_csv = DataFrame(c=Float64[], k_act=Float64[], inv_alpha_act=Float64[], k_obs=Float64[], inv_alpha_obs=Float64[])

try
  for i in 1:max_iters
      sims = []
      params = Vector{Float64}[]
      print("creating $K simulation sets")
      for k in 1:K
          p = rand(d)
          p[1] = max(0.0, p[1])
          p[2] = max(1.0, p[2])
          p[3] = max(0.1, p[3])
          p[4] = max(1.0, p[4])
          p[5] = max(0.1, p[5])
          push!(params, p)
          k_sims = gen_sims(p, n, k, i)
          print(".")
          append!(sims, k_sims)
      end
      @assert length(params) == K
      println()
      if parallelize
        results = run_parallel(sims)
      else
        results = run(sims)
      end
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
              println("pos def exception")
              global d = fit(typeof(d), elite_matrix += 0.01*randn(size(elite_matrix)))
          else
              rethrow(ex)
          end
      end
      println("iteration $i")
      @show mean(d)
      @show det(cov(d))
      @show ev = eigvals(cov(d))
      for j in 1:length(ev)
          @show eigvecs(cov(d))[:,j]
      end
      push!(results_csv, mean(d))
      if save_data
        fname_csv = joinpath(@__DIR__, "../data", "LQG_pomcpow_opt"*"_"*datestring*".csv")
        CSV.write(fname_csv, results_csv)
        print("\nSaving parameters to ", fname_csv, "\n")
      end
  end
finally
  Distributed.rmprocs(worker_ids)
end

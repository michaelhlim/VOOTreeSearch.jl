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

  using MCTS
  using StaticArrays
  using POMDPModelTools: GenerativeBeliefMDP
  include("./scripts/VDPTag2/vdp_next_action.jl")
end

@info "Started $(Distributed.nworkers()) workers..."
@info "Precompiling simulation code..."

max_time = 1.0
max_depth = 10

@show start_mean = [110.0, 30.0, 30.0, 5.0, 100.0, 0.8]
@show start_cov = diagm([100.0^2, 10.0^2, 10.0^2, 10.0^2, 20.0^2, 0.1^2])
d = MvNormal(start_mean, start_cov)
on_cluster = true
save_data = true
parallelize = true

if on_cluster
  max_query = 100_000
  K = 150 # number of parameter samples
  n = 20 # number of evaluation simulations
  m = 30 # number of elite samples
  max_iters = 50
else
  max_query = 1000
  K = 15 # number of parameter samples
  n = 1 # number of evaluation simulations
  m = 10 # number of elite samples
  max_iters = 3 
end


# VDPTag2
@show pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))

println(stderr, "Max time: ", max_time)
println(stderr, "Max depth: ", max_depth)
println(stderr, "Max query: ", max_query)
println(stderr, "Start mean: ", start_mean)
println(stderr, "Start cov: ", start_cov)
println(stderr, "POMDP: ", pomdp)

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
    p_voo = x[6]
    @assert p_voo >= 0.1
    @assert p_voo <= 1.0

    sims = []

    for i in 1:n
        rng = MersenneTwister(7)

        ro = ToNextMLSolver(rng)
        sig = diagm([0.1])
        epsilon = 0.1/10.0
        voo_opt = VOOActionGenerator(p_voo, Distributions.Uniform(), MvNormal(sig), rng, epsilon)

        solver = POMCPOWSolver(tree_queries=max_query,
                               criterion=MaxUCB(c),
                               max_depth=max_depth,
                               max_time=max_time,
                               k_action=k_act,
                               alpha_action=1/inv_alpha_act,
                               k_observation=k_obs,
                               alpha_observation=1/inv_alpha_obs,
                               estimate_value=FORollout(ro),
                               next_action = RootToNextMLFirstVOO(voo_opt),
                               default_action=BasicPOMCP.ReportWhenUsed(TagAction(false, 0.0)),
                               rng=rng
                              )

        planner = solve(solver, pomdp)
        filter = SIRParticleFilter(deepcopy(pomdp), 
                                   100_000, 
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
results_csv = DataFrame(c=Float64[], k_act=Float64[], inv_alpha_act=Float64[], k_obs=Float64[], inv_alpha_obs=Float64[], p=Float64[])
try
  for i in 1:max_iters
      sims = []
      params = Vector{Float64}[]
      println(stderr,"creating $K simulation sets")
      for k in 1:K
          p = rand(d)
          p[1] = max(0.0, p[1])
          p[2] = max(1.0, p[2])
          p[3] = max(0.1, p[3])
          p[4] = max(1.0, p[4])
          p[5] = max(0.1, p[5])
          p[6] = max(0.1, min(1.0, p[6]))
          push!(params, p)
          k_sims = gen_sims(p, n, k, i)
          append!(sims, k_sims)
      end
      @assert length(params) == K
      if parallelize
        results = run_parallel(sims)
      else
        results = run(sims)
      end
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
      println(stderr, "Cov (det): ", det(cov(d)))
      ev = eigvals(cov(d))
      println(stderr, "Cov (eig): ", ev)
      for j in 1:length(ev)
          println(stderr, "Eigvecs: ", eigvecs(cov(d))[:,j])
      end
      push!(results_csv, mean(d))
      if save_data
        fname_csv = joinpath(@__DIR__, "../data", "VDP_vomcpow_opt"*"_"*datestring*".csv")
        CSV.write(fname_csv, results_csv)
        println(stderr,"Saving parameters to ", fname_csv)
      end
  end
finally
  Distributed.rmprocs(worker_ids)
end

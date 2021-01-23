##
# sparse sampling solver environments

abstract type AbstractVSSOptions end

# accessors
maxdepth(opt::AbstractVSSOptions) = opt.maxdepth
width(opt::AbstractVSSOptions) = opt.width
action_width(opt::AbstractVSSOptions) = opt.action_width

@with_kw struct VOSSOptions{V<:VOOActionGenerator} <: AbstractVSSOptions
    maxdepth::Int
    width::Int
    action_width::Int
    action_width_decay::Float64
    last_action_null::Bool
    voo::V
end

@with_kw struct VOWSSOptions{V<:VOOActionGenerator} <: AbstractVSSOptions
    maxdepth::Int
    width::Int
    action_width::Int
    action_width_decay::Float64
    last_action_null::Bool
    voo::V
end

struct VOOSparseSamplingSolver
    opt::AbstractVSSOptions
    rng::AbstractRNG
end

VOOSparseSamplingSolver(opt::AbstractVSSOptions; rng=Random.GLOBAL_RNG) = VOOSparseSamplingSolver(opt, rng)
VOOSparseSamplingSolver(;rng=Random.GLOBAL_RNG, kwargs...) = VOOSparseSamplingSolver(VOWSSOptions(kwargs...), rng)

function POMDPs.solve(s::VOOSparseSamplingSolver, m::Union{MDP,POMDP})
    if s.opt isa VOWSSOptions
        @assert !(Nothing <: statetype(m)) "POWSS does not support problems where the state can be nothing (this can be fixed by adding Some in appropriate places)."
    end
    VSSPlanner(m, s.opt, s.rng)
end

struct VSSPlanner{M<:Union{MDP,POMDP},O<:AbstractVSSOptions,R<:AbstractRNG}
    m::M
    opt::O
    rng::R
end
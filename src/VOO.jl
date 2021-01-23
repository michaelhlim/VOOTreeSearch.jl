##
# VOO Routines

struct VOOActionGenerator{D1 <: Distribution, D2 <: Distribution, R <: AbstractRNG} <: Function
	exploration_prob::Float64
	exploration_sampler::D1
    voronoi_sampler::D2
    action_rng::R
    acceptance_radius::Float64
    early_halt::Bool
    halt_count::Int64
end

# If only using acceptance radius
function VOOActionGenerator(exploration_prob::Float64,
                            exploration_sampler::Distribution,
                            voronoi_sampler::Distribution,
                            action_rng::AbstractRNG,
                            acceptance_radius::Float64)
    early_halt = false
    halt_count = 1
    return VOOActionGenerator(exploration_prob, exploration_sampler, voronoi_sampler, action_rng, acceptance_radius, early_halt, halt_count)
end

# If only using halt count
function VOOActionGenerator(exploration_prob::Float64,
                            exploration_sampler::Distribution,
                            voronoi_sampler::Distribution,
                            early_halt::Bool,
                            halt_count::Int64)
    acceptance_radius = 0.0
    return VOOActionGenerator(exploration_prob, exploration_sampler, voronoi_sampler, action_rng, acceptance_radius, early_halt, halt_count)
end

function voo_sample(f::VOOActionGenerator, a_list, q_list, a_space)
	# determine explore vs. exploit
	w = rand(f.exploration_sampler)
	
	if w <= f.exploration_prob || length(a_list) <= 1
		# uniform sample from action space
		a_new = rand(f.action_rng, a_space)
	else
	    # find the best value
	    a_new = voronoi_sample_centered(f, a_list, q_list, a_space)
	end
    return a_new
end

function voo_sample(f::VOOActionGenerator, problem::Union{MDP,POMDP}, b, h::DPWStateNode)
    # extract actions
    t = h.tree
    a_inds = t.children[h.index]
    a_list = t.a_labels[a_inds]
    a_space = actions(problem, b)

    # determine explore vs. exploit
    w = rand(f.exploration_sampler)
    
    if w <= f.exploration_prob || length(a_list) <= 1
        # uniform sample from action space
        a_new = rand(f.action_rng, a_space)
    else
        # extract q values
        q_list = [t.q[a] for a in a_inds]

        # find the best value
        a_new = voronoi_sample_centered(f, a_list, q_list, a_space)
    end
    return a_new
end

function voo_sample(f::VOOActionGenerator, problem::Union{MDP,POMDP}, b, h::POWTreeObsNode)
	# extract actions
	t = h.tree
    a_inds = t.tried[h.node]
    a_list = t.a_labels[a_inds]
    a_space = actions(problem, b)

    # determine explore vs. exploit
	w = rand(f.exploration_sampler)
	
	if w <= f.exploration_prob || length(a_list) <= 1
		# uniform sample from action space
		a_new = rand(f.action_rng, a_space)
	else
	    # extract q values
		q_list = [t.v[a] for a in a_inds]

	    # find the best value
	    a_new = voronoi_sample_centered(f, a_list, q_list, a_space)
	end
    return a_new
end

function voronoi_sample_centered(f::VOOActionGenerator, a_list, q_list, a_space)
	best_a = a_list[argmax(q_list)]
	a_new = best_a
    closest = false

    a_closest = best_a
    dist_closest = Inf

    # iterate until it lies within the best Voronoi cell
    n = 1
    while .!closest
    	closest = true
        a_new = action_sample_centered(best_a, f.voronoi_sampler, a_space)
        dist_to_best = action_dist(a_new, best_a)

        if dist_to_best < dist_closest
            a_closest = a_new
            dist_closest = dist_to_best
        end

        # if early halting, return the closest one to the center
        if f.early_halt && n >= f.halt_count
            return a_closest
        end

        # if auto acceptance radius, return if the sample is within the radius
        if dist_to_best < f.acceptance_radius
        	return a_new
        end

        # iterate over all actions to see if it is closer to another Voronoi cell
        for a in a_list
        	if action_dist(a_new, a) < dist_to_best
        		closest = false
        	end
        end
        n += 1
    end
    return a_new
end

function (f::VOOActionGenerator)(pomdp, b, h)
    a_new = voo_sample(f, pomdp, b, h)
    return a_new
end


# Next action function that uses the rollout policy as the first action
abstract type AbstractRolloutFirst end

struct RolloutFirst{R <: AbstractRNG, P <: Policy} <: AbstractRolloutFirst
    rng::R
    rollout_policy::P
end

function MCTS.next_action(gen::RolloutFirst, p::POMDP, b, h)
    if isroot(h) && n_children(h) < 1
        return POMDPs.action(gen.rollout_policy, b)
    else
        return rand(gen.rng, actions(p))
    end
end

MCTS.next_action(gen::RolloutFirst, p::GenerativeBeliefMDP, b, h) = next_action(gen, p.pomdp, b, h)

struct VOORolloutFirst{V <: VOOActionGenerator, P <: Policy} <: AbstractRolloutFirst
    voo::V
    rollout_policy::P
end

function MCTS.next_action(gen::VOORolloutFirst, p::POMDP, b, h)
    if isroot(h) && n_children(h) < 1
        return POMDPs.action(gen.rollout_policy, b)
    else
        return voo_sample(gen.voo, p, b, h)
    end
end

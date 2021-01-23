##
# New environments

struct BoxND
	# create one with e.g. BoxND(2, [0.0, 0.0], [1.0, 1.0])
	dim::Int64
    min::Vector{Float64}
    max::Vector{Float64}
end

function Base.rand(rng::AbstractRNG, s::Random.SamplerTrivial{BoxND})
	box = s[]
	v_range = box.max - box.min

	return v_range .* rand(rng, box.dim) + box.min
end

##
# Different definitions for action distances and proposals

# Box actions
function action_dist(a1::Vector{Float64}, a2::Vector{Float64})
	return euclidean(a1, a2)
end

function action_sample_centered(best_a::Vector{Float64}, sampler, a_space::BoxND)
	in_bounds = false
	a_new = Vector{Float64}

	# make sure new sample is in bounds
	while !in_bounds
		a_new = rand(sampler) + best_a
		in_bounds = minimum(a_new .> a_space.min) && minimum(a_new .< a_space.max)
	end

	return a_new
end

# TagAction hybrid actions
function action_dist(a1::TagAction, a2::TagAction)
	if a1.look == a2.look
		angle_diff = euclidean(a1.angle, a2.angle)
		return minimum([angle_diff, 2 * pi - angle_diff])
	else	
		return Inf
	end
end

function action_sample_centered(best_a::TagAction, sampler, a_space)
	look_temp = best_a.look
	angle_temp = mod(rand(sampler)[1] + best_a.angle, 2 * pi)
	return TagAction(look_temp, angle_temp)
end

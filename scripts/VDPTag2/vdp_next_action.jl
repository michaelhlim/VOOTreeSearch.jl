struct RootToNextMLFirstVOO
    voo::VOOActionGenerator
end

function MCTS.next_action(gen::RootToNextMLFirstVOO, p::VDPTagPOMDP, b, node)
    if isroot(node) && n_children(node) < 1
        target_sum=MVector(0.0, 0.0)
        agent_sum=MVector(0.0, 0.0)
        for s in particles(b::ParticleCollection)
            target_sum += s.target
            agent_sum += s.agent
        end
        next = VDPTag2.next_ml_target(mdp(p), target_sum/n_particles(b))
        diff = next-agent_sum/n_particles(b)
        return TagAction(false, atan(diff[2], diff[1]))
    else
        return voo_sample(gen.voo, p, b, node)
    end
end

MCTS.next_action(gen::RootToNextMLFirstVOO, p::GenerativeBeliefMDP, b, node) = next_action(gen, p.pomdp, b, node)
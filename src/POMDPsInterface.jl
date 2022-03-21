function reward(m::POMDP{S}, s::S, a::CovidAction, sp::S) where S <: CovidState
    inf_loss = m.inf_loss*infected(sp)/m.N
    test_loss = m.test_loss*a.testing_prop
    testrate_loss = m.testrate_loss*abs(a.testing_prop-s.prev_action.testing_prop)
    return -(inf_loss + test_loss + testrate_loss)
end

function continuous_gen(m::POMDP{S}, s::S, a::CovidAction, rng::AbstractRNG=Random.GLOBAL_RNG) where S <: CovidState
    sp, new_inf, o = sim_step(m, s, a)
    r = reward(m, s, a, sp)
    o = sum(o)

    return (sp=sp, o=o, r=r)
end

function POMDPs.gen(pomdp::POMDP{S}, s::S, a::CovidAction, rng::AbstractRNG=Random.GLOBAL_RNG) where S <: CovidState
    rsum = 0.0
    local o::Int
    for i in 1:pomdp.test_period
        s,o,r = continuous_gen(pomdp, s, a, rng)
        rsum += r
    end
    return (sp=s, o=o, r=rsum)
end


#=
TODO: Find type stable way to do this with Normal() ?
Currently, zero testing prop results in obersvation distribution of Normal(0,0),
resulting in NaNs for weights when normalizing.
=#
struct UninformedDist end

Distributions.pdf(::UninformedDist, ::Any) = 1.0

POMDPs.actions(pomdp::AbstractCovidPOMDP) = pomdp.actions

POMDPs.discount(pomdp::AbstractCovidPOMDP) = pomdp.discount

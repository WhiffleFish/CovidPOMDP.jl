struct ProportionalControlSolver
    k::Float64
end

ProportionalControlSolver() = ProportionalControlSolver(0.5)

struct ProportionalControlPlanner
    k::Float64
end

function BasicPOMCP.convert_estimator(pc::ProportionalControlSolver, ::Any, pomdp::CovidPOMDP)
    return ProportionalControlPlanner(pc.k)
end

function POMDPs.action(pc::ProportionalControlPlanner, s::CovidState)
    I = sum(s.I)
    inf_prop = I / (s.S + I + s.R)
    return CovidAction(pc.k*inf_prop)
end

function BasicPOMCP.estimate_value(
    estimator::ProportionalControlPlanner,
    pomdp::POMDP{S},
    s::S,
    node,
    depth::Int) where {S}

    disc = 1.0
    r_total = 0.0
    step = 0
    γ = discount(pomdp)

    while !isterminal(pomdp, s) && step < depth

        a = action(estimator, s)

        sp,o,r = gen(pomdp, s, a)

        r_total += disc*r

        s = sp

        disc *= γ
        step += 1
    end

    return r_total
end

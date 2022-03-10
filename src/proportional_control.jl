struct ProportionalControlSolver <: Solver
    k::Float64
end

ProportionalControlSolver() = ProportionalControlSolver(1.0)

struct ProportionalControlPlanner <: Policy
    k::Float64
end

function POMDPs.solve(pc::ProportionalControlSolver, pomdp::CovidPOMDP)
    return ProportionalControlPlanner(pc.k)
end

function BasicPOMCP.convert_estimator(pc::ProportionalControlSolver, ::Any, pomdp::CovidPOMDP)
    return solve(pc, pomdp)
end

function POMDPs.action(pc::ProportionalControlPlanner, s::CovidState)
    I = sum(s.I)
    inf_prop = I / (s.S + I + s.R)
    return CovidAction(min(pc.k*inf_prop, 1.0))
end

function POMDPs.action(pc::ProportionalControlPlanner, b::ParticleCollection{CovidState})
    s̄ = Statistics.mean(b.particles)
    return POMDPs.action(pc, s̄)
end

function POMDPs.action(pc::ProportionalControlPlanner, b)
    b = ParticleCollection([rand(b) for _ in 1:100]) # TODO: change default particle size
    return POMDPs.action(pc, b)
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

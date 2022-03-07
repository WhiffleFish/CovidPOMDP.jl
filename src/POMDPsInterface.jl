function POMDPs.gen(pomdp::CovidPOMDP, s::CovidState, a::CovidAction, rng::AbstractRNG=Random.GLOBAL_RNG)
    rsum = 0.0
    local o::obstype(pomdp)
    for i in 1:pomdp.test_period
        s,o,r = continuous_gen(pomdp, s, a, rng)
        rsum += r
    end
    return (sp=s, o=o, r=rsum)
end

function POMDPs.observation(pomdp::CovidPOMDP, s::CovidState, a::CovidAction, sp::CovidState)
    tot_mean = 0.0
    tot_variance = 0.0
    p = s.params
    for (i,inf) in enumerate(s.I)
        num_already_tested = sum(@view s.Tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,p.pos_test_probs[i])
        tot_mean += Statistics.mean(dist)
        tot_variance += Statistics.var(dist)
    end
    return Normal(tot_mean, sqrt(tot_variance))
end

POMDPs.actions(pomdp::CovidPOMDP) = pomdp.actions

POMDPs.discount(pomdp::CovidPOMDP) = pomdp.discount

function POMDPs.simulate(
    pomdp::CovidPOMDP,
    b,
    planner::Policy;
    s = rand(b),
    T::Int=50,
    n_p::Int = 100,
    progress::Bool=false)

    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(CovidAction,T)
    rewardHist = zeros(Float64,T)
    beliefHist = Vector{ParticleCollection{CovidState}}(undef, T)

    single_step_pomdp = unity_test_period(pomdp)
    upd = BootstrapFilter(single_step_pomdp, n_p)

    prog = Progress(T; enabled=progress)
    for day in 1:T

        if (day-1)%pomdp.test_period == 0
            a = POMDPs.action(planner, b)
        else
            a = actionHist[day-1]
        end
        (day == 1) && (b = initialize_belief(upd, b))

        susHist[day] = s.S
        infHist[day] = sum(s.I)
        recHist[day] = s.R
        actionHist[day] = a
        beliefHist[day] = b

        s, o, r = POMDPs.gen(single_step_pomdp, s, a)
        b = update(upd, b, a, o)

        rewardHist[day] = r
        testHist[day] = sum(o)

        next!(prog)
    end
    return SimHist(susHist, infHist, recHist, pomdp.N, T, testHist, actionHist, rewardHist, beliefHist)
end

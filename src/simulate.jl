"""
# Arguments
- `pomdp::CovidPOMDP` - Simulation parameters
- `state::CovidState` - Current Sim State
- `action::CovidAction` - Current Sim Action
- `T::Int` - Simulation duration (days)
"""
function POMDPs.simulate(
    pomdp::POMDP{S},
    state::S = S(pomdp),
    action::CovidAction = CovidAction(0.0);
    T::Int = 50
    ) where S <: CovidState

    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = fill(action, T)
    rewardHist = zeros(Float64, T)

    for day in 1:T
        susHist[day] = state.S
        infHist[day] = infected(state)
        recHist[day] = state.R

        sp, new_infections, pos_tests = sim_step(pomdp, state, action)
        r = reward(pomdp, state, action,sp)

        testHist[day] = sum(pos_tests)
        rewardHist[day] = r
        state = sp
    end

    return SimHist(
        susHist,
        infHist,
        recHist,
        pomdp.N,
        T,
        testHist,
        actionHist,
        rewardHist,
        ParticleCollection{S}[]
    )
end

function POMDPs.simulate(
    pomdp::POMDP{S},
    b,
    planner::Policy;
    s = rand(b),
    T::Int=50,
    upd = BootstrapFilter,
    n_p::Int = 10_000,
    progress::Bool=false) where S <: CovidState

    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = zeros(CovidAction,T)
    rewardHist = zeros(Float64,T)
    beliefHist = Vector{ParticleCollection{S}}(undef, T)

    single_step_pomdp = unity_test_period(pomdp)
    upd = upd(single_step_pomdp, n_p)

    prog = Progress(T; enabled=progress)
    for day in 1:T

        if (day-1)%pomdp.test_period == 0
            a = POMDPs.action(planner, b)
        else
            a = actionHist[day-1]
        end
        (day == 1) && (b = initialize_belief(upd, b))

        susHist[day] = s.S
        infHist[day] = infected(s)
        recHist[day] = s.R
        actionHist[day] = a
        beliefHist[day] = b

        s, o, r = POMDPs.gen(single_step_pomdp, s, a)
        b = update(upd, b, a, o)

        rewardHist[day] = r
        testHist[day] = o

        next!(prog)
    end
    return SimHist(susHist, infHist, recHist, pomdp.N, T, testHist, actionHist, rewardHist, beliefHist)
end

@inline function shift_inf!(I::Vector)
    @inbounds for i in length(I):-1:2
        I[i] = I[i-1]
    end
    I[1] = 0
    I
end

"""
Progress testing state forward
People k days from receiving test back are now k-1 days from receiving test
Tested individuals with infection age t move to infection age t + 1
"""
@inline function shift_test!(T::Matrix)
    s1,s2 = size(T)
    @inbounds for i in 1:(s1-1)
        @views T[i,:] .= T[i+1,:]
    end
    @inbounds for j in s2:-1:2
        @views T[:,j] .= T[:,j-1]
    end
    T[:,1] .= 0
    T[end,:] .= 0
    return T
end

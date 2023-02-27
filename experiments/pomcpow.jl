using POMCPOW
using CovidPOMDPs
pomdp = CovidPOMDP(
    inf_loss = 10.,
    actions = CovidPOMDPs.CovidActionSpace(7;zero=true)
)
b0 = initialstate(pomdp)

sol = POMCPOWSolver(
    criterion = MaxUCB(2.0),
    max_depth = 30,
    tree_queries = 10_000,
    check_repeat_act = true,
    check_repeat_obs = false,
    enable_action_pw = false,
    # k_action = 5,
    # alpha_action = 0.1,
    k_observation = 3,
    alpha_observation = 0.05,
    estimate_value = CovidPOMDPs.ProportionalControlSolver(10.0)
)
planner = solve(sol, pomdp)
@time action(planner, b0)

using ProgressMeter
@progress v_1000 = [action(planner, b0) for i in 1:100]
hist(getfield.(v_1000, :testing_prop))

@progress v_10_000 = [action(planner, b0) for i in 1:100]
hist(getfield.(v_10_000, :testing_prop))

@progress v_50_000 = [action(planner, b0) for i in 1:100]
hist(
    getfield.(v_50_000, :testing_prop),
    bins=7,
    axis = (xticks=LinearTicks(7),)
)

using D3Trees
D3Tree(planner.tree) |> inchrome

b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))

h = simulate(pomdp, b0, planner; T=100, upd=CovidPOMDPs.NoisyCovidFilter, n_p = 10_000, progress=true)
lines(
    getproperty.(h.actions, :testing_prop),
    linewidth=4,
    axis = (;xlabel="day", ylabel="Testing Proportion"))

lines(h.rewards; linewidth=2)

plot(h)

CovidPOMDPs.plot_inf_belief(h)

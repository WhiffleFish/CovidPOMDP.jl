using CovidPOMDPs
using POMCPOW

pomdp = SingleCovidPOMDP(inf_loss=10.0, actions=CovidPOMDPs.CovidActionSpace(7;zero=true))
b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))
s0 = rand(b0)

prop_sol = CovidPOMDPs.ProportionalControlSolver(10.0)
prop_planner = solve(prop_sol, pomdp)

pomcp_sol = POMCPSolver(
    tree_queries    = 10_000,
    max_depth       = 30,
    c               = 2.0,
    estimate_value  = CovidPOMDPs.ProportionalControlSolver(10.0),
)
pomcp_planner = solve(pomcp_sol, pomdp)

pomcpow_sol = POMCPOWSolver(
    tree_queries        = 10_000,
    max_depth           = 30,
    criterion           = MaxUCB(2.0),
    check_repeat_act    = true,
    check_repeat_obs    = false,
    enable_action_pw    = false,
    k_observation       = 3,
    alpha_observation   = 0.05,
    estimate_value      = CovidPOMDPs.ProportionalControlSolver(10.0)
)
pomcpow_planner = solve(pomcpow_sol, pomdp)

T = 400

@info "PROPORTIONAL"
h_prop = simulate(
    pomdp,
    b0,
    prop_planner;
    s = s0,
    T = T,
    upd = CovidPOMDPs.NoisyCovidFilter,
    n_p = 10_000,
    progress = true
)

@info "POMCP"
h_pomcp = simulate(
    pomdp,
    b0,
    pomcp_planner;
    s = s0,
    T = T,
    upd = CovidPOMDPs.NoisyCovidFilter,
    n_p = 10_000,
    progress = true
)

@info "POMCPOW"
h_pomcpow = simulate(
    pomdp,
    b0,
    pomcpow_planner;
    s = s0,
    T = T,
    upd = CovidPOMDPs.NoisyCovidFilter,
    n_p = 10_000,
    progress = true
)

@info "SAVE"
using JLD2
save_path = joinpath("D:\\", "CovidPOMDP", "SingleCovid.jld2")
save(save_path, Dict("h_prop"=>h_prop, "h_pomcp"=>h_pomcp,"h_pomcpow"=>h_pomcpow))

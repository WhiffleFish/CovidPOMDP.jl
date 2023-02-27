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

@profiler action(pomcpow_planner, b0)

T = 400

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

plot(h_pomcpow)
plot(h_pomcp)
plot(h_prop)

begin
    kwargs = (;linewidth=2)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Day", ylabel="Cumulative Reward")
    lines!(ax, cumsum(h_pomcpow.rewards); label="POMCPOW", kwargs...)
    lines!(ax, cumsum(h_pomcp.rewards); label="POMCP", kwargs...)
    lines!(ax, cumsum(h_prop.rewards); label="Proportional", kwargs...)
    axislegend(ax)
    fig
end
save("reward_compare.pdf", fig)

begin
    kwargs = (;linewidth=2)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Day", ylabel="Testing Proportion")
    lines!(ax, getproperty.(h_pomcpow.actions, :testing_prop); label="POMCPOW", kwargs...)
    lines!(ax, getproperty.(h_pomcp.actions, :testing_prop); label="POMCP", kwargs...)
    lines!(ax, getproperty.(h_prop.actions, :testing_prop); label="Proportional", kwargs...)
    axislegend(ax)
    fig
end
save("action_compare.pdf", fig)

CovidPOMDPs.plot_inf_belief(h_prop; particle_samples=100, alpha=0.01)

fig = CovidPOMDPs.plot_inf_belief(h_prop; particle_samples=500, axis=(;xlabel="Day", ylabel="Infected Population Proportion"))
CovidPOMDPs.plot_inf_belief(h_pomcp)
CovidPOMDPs.plot_inf_belief(h_pomcpow)

using JLD2
save("round1.jld2", Dict("h_prop"=>h_prop, "h_pomcp"=>h_pomcp,"h_pomcpow"=>h_pomcpow))
h_pomcp


##
using CovidPOMDPs
using JLD2
path = joinpath("D:\\", "CovidPOMDP", "SingleCovid.jld2")
d = load(path)
h_pomcpow = d["h_pomcpow"]
h_pomcp = d["h_pomcp"]
h_prop = d["h_prop"]
CovidPOMDPs.plot_inf_belief(h_pomcpow; particle_samples=100)
CovidPOMDPs.plot_inf_belief(h_pomcp; particle_samples=100)
fig = CovidPOMDPs.plot_inf_belief(h_prop; particle_samples=100)
save("PropBelief_small.pdf", fig)

pps = particles.(h_pomcpow.beliefs)



begin
    idx = 50
    inf_idx = 4
    ps = pps[idx]
    ps_params = getfield.(ps, :params)
    ps_inf = getfield.(ps_params, :infectiousness)
    ps_params = [params(p[inf_idx]) for p in ps_inf]
    scatter(Iterators.repeat(1:1, 10_000), first.(ps_params); axis=(;limits=(nothing, nothing, nothing, nothing)))
end

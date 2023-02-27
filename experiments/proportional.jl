using CovidPOMDPs

pomdp = SingleCovidPOMDP()
sol = CovidPOMDPs.ProportionalControlSolver(10.0)
planner = solve(sol, pomdp)

b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))

h = simulate(pomdp, b0, planner; T=200, upd=CovidPOMDPs.NoisyCovidFilter, n_p = 10_000, progress=true)

plot(h)
lines(
    getproperty.(h.actions, :testing_prop),
    linewidth = 4,
    axis = (;xlabel="day", ylabel="Testing Proportion"))

CovidPOMDPs.plot_inf_belief(h)


##
pomdp = DoubleCovidPOMDP()
sol = CovidPOMDPs.ProportionalControlSolver(10.0)
planner = solve(sol, pomdp)

b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))

h = simulate(pomdp, b0, planner; T=100, upd=CovidPOMDPs.NoisyCovidFilter, n_p = 10_000, progress=true)

plot(h)
lines(
    getproperty.(h.actions, :testing_prop),
    linewidth = 4,
    axis = (;xlabel="day", ylabel="Testing Proportion"))

CovidPOMDPs.plot_inf_belief(h)

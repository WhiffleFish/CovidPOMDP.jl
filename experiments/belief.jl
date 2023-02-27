using CovidPOMDPs

pomdp = CovidPOMDP()

b0 = initialstate(pomdp, Distributions.Uniform(1,10_000))

s0 = rand(b0)
CovidPOMDPs.SIR(s0)

struct FunctionPolicy{F} <: Policy
    f::F
end

POMDPs.action(p::FunctionPolicy, b::Any) = p.f(b)

fp = FunctionPolicy(b->CovidAction(0.10))

hist = simulate(pomdp, b0, fp; T=100, upd=CovidPOMDPs.NoisyCovidFilter, n_p = 10_000, progress=true)
fig = plot(hist)
save(joinpath(@__DIR__, "img", "stackplot.svg"), fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    CovidPOMDPs.plot_inf_belief!(ax, hist, alpha=5e-3)
    display(fig)
end

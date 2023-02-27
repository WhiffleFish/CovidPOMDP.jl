using CovidPOMDPs

struct FunctionPolicy{F} <: Policy
    f::F
end

POMDPs.action(p::FunctionPolicy, b::Any) = p.f(b)

fp = FunctionPolicy(b->CovidAction(0.10))

pomdp = DoubleCovidPOMDP()

b0 = initialstate(pomdp, Distributions.Uniform(1,50_000))

s0 = rand(b0)
CovidPOMDPs.SIR(s0)

h,Ness = simulate(pomdp, b0, fp; s=s0, T=100, upd=CovidPOMDPs.NoisyCovidFilter, n_p = 10_000, progress=true, ret_Ness=true)
h,Ness = simulate(pomdp, b0, fp; s=s0, T=100, upd=BootstrapFilter, n_p = 10_000, progress=true, ret_Ness=true)
fig = plot(h)
replace!(Ness) do x
    isnan(x) ? 0.0 : x
end
save(joinpath(@__DIR__, "img", "stackplot.svg"), fig)

with_theme(Theme(fontsize = 20)) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_{ess}", fontsize=20)
    lines!(ax, Ness, linewidth=3)
    save("DoubleStrain_SIR_Ness.pdf",fig)
    fig
end



begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    CovidPOMDPs.plot_inf_belief!(ax, h, alpha=1e-2)
    display(fig)
    save("DoubleStrain_SIR_Particles.pdf", fig)
end

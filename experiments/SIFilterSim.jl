using CovidPOMDPs
using CovidPOMDPs: SICovidFilter, unity_test_period, CovidState
using ProgressMeter

function SIsimulate(
    pomdp::POMDP{S},
    b,
    planner::Policy;
    s = rand(b),
    T::Int=50,
    upd = SICovidFilter,
    n_p::Int = 10_000,
    progress::Bool = false,
    ret_Ness::Bool = true) where S <: CovidState

    stateHist = Vector{S}(undef, T)
    actionHist = zeros(CovidAction, T)
    rewardHist = zeros(Float64, T)
    testHist = Vector{Int}(undef, T)
    beliefHist = Matrix{S}(undef, n_p, T)
    weightHist = Matrix{Float64}(undef, n_p, T)
    ret_Ness && (Ness = Vector{Float64}(undef, T))

    single_step_pomdp = unity_test_period(pomdp)
    upd = upd(single_step_pomdp, n_p)

    prog = Progress(T; enabled=progress)
    for day in 1:T
        if (day-1)%pomdp.test_period == 0
            a = POMDPs.action(planner, b)
        else
            a = actionHist[day-1]
        end
        isone(day) && (b = initialize_belief(upd, b))

        stateHist[day] = s
        actionHist[day] = a
        beliefHist[:,day] .= b.particles
        weightHist[:,day] .= upd.weights
        ret_Ness && (Ness[day] = CovidPOMDPs.effective_size(weights(upd)))

        s, o, r = POMDPs.gen(single_step_pomdp, s, a)
        b = update(upd, b, a, o)

        rewardHist[day] = r
        testHist[day] = o

        next!(prog)
    end

    # hist = SimHist(stateHist, pomdp.N, T, testHist, actionHist, rewardHist, beliefHist)
    return stateHist, beliefHist, weightHist, Ness
end

function _vectorize_inf(ps::Matrix{S}, pop::Int) where S <: CovidState
    T = size(ps, 2)
    Np = size(ps, 1)
    x = Vector{Int}(undef, Np*T)
    y = Vector{Float64}(undef, Np*T)

    for (i,ci) in enumerate(CartesianIndices((Np,T)))
        p,t = Tuple(ci)
        x[i] = t
        s = ps[p,t]
        y[i] = infected(s) / pop
    end
    return x, y
end

function plot_inf_belief!(ax::Axis, stateHist, beliefHist, weightHist; alpha::Float64=0.01)
    pop = CovidPOMDPs.population(first(stateHist))
    T = length(stateHist)
    lines!(ax, infected.(stateHist) ./ pop, linewidth=4)
    x,y = _vectorize_inf(beliefHist, pop)
    colors = [(:red, x*2) for x in weightHist]
    scatter!(
        ax,
        x,
        y,
        color=colors
    )

    return ax
end

struct FunctionPolicy{F} <: Policy
    f::F
end

POMDPs.action(p::FunctionPolicy, b::Any) = p.f(b)

fp = FunctionPolicy(b->CovidAction(0.10))

pomdp = DoubleCovidPOMDP()

b0 = initialstate(pomdp, Distributions.Uniform(1,50_000))

s0 = rand(b0)
CovidPOMDPs.SIR(s0)

statehist, beliefHist, weightHist, Ness = SIsimulate(pomdp, b0, fp, ;s=s0, T=100, upd=SICovidFilter, n_p = 10_000, progress=true)

fig = Figure()
ax = Axis(fig[1,1])
plot_inf_belief!(ax, statehist, beliefHist, weightHist)
fig
save("DoubleStrain_SIS_Particles.pdf",fig)

replace!(Ness) do x
    isnan(x) ? 0.0 : x
end

with_theme(Theme(fontsize = 20)) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_{ess}", fontsize=20)
    lines!(ax, Ness, linewidth=3)
    save("DoubleStrain_SIS_Ness.pdf",fig)
    fig
end

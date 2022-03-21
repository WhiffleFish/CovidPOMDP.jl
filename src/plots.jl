function CairoMakie.plot(hist::SimHist; prop::Bool=true, kind::Symbol=:stack, kwargs...)
    if kind === :stack
        return stackplot(hist; kwargs...)
    elseif kind === :line
        return lineplot(hist; kwargs...)
    else
        throw(DomainError("`kind` must be either `:stack` or `:line`"))
    end
end

function stackplot(hist::SimHist; kwargs...)
    N = hist.N
    S = hist.sus ./ N
    I = hist.inf ./ N
    R = hist.rec ./ N
    T = hist.T

    CS = S .+ I
    CR = CS .+ R
    band_α = 0.5

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel="Day",
        ylabel="Population Proportion";
        kwargs...
    )
    lines!(ax, 1:T, I, color="#E69F00", linewidth=3)
    bi = band!(ax, 1:T, fill(0,T), I, color=("#E69F00", band_α), label="Infected")

    lines!(ax, 1:T, CS, color="#56B4E9", linewidth=3)
    bs = band!(ax, 1:T, I, CS, color=("#56B4E9", band_α), label="Susceptible")

    lines!(ax, 1:T, CR, color="#009E73", linewidth=3)
    br = band!(ax, 1:T, CS, CR, color=("#009E73", band_α), label="Recovered")

    limits!(ax, 1, T, 0, 1.0)
    axislegend(ax, [bs, bi, br], ["Susceptible", "Infectious", "Recovered"])
    return fig
end


function lineplot(hist::SimHist; kwargs...)
    N = hist.N
    S = hist.sus ./ N
    I = hist.inf ./ N
    R = hist.rec ./ N
    T = hist.T

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel="Day",
        ylabel="Population Proportion";
        kwargs...)

    lines!(ax, 1:T, S, linewidth=3, label="Susceptible")
    lines!(ax, 1:T, I, linewidth=3, label="Infectious")
    lines!(ax, 1:T, R, linewidth=3, label="Recovered")
    limits!(ax, 1, T, 0, 1.0)
    axislegend(ax)
    return fig
end

function plot_inf_belief!(ax::Axis, hist::SimHist; particle_samples=0, alpha::Float64=0.01)
    pop = hist.N
    T = hist.T
    lines!(ax, hist.inf ./ pop, linewidth=4)
    if particle_samples ≤ 0
        x,y = _vectorize_inf(hist.beliefs, pop)
        scatter!(
            ax,
            x,
            y,
            color=(:red, alpha)
        )
    else
        x,y = _vectorize_inf_sample(hist.beliefs, pop, particle_samples)
        scatter!(
            ax,
            x,
            y,
            color=(:red, alpha)
        )
    end

    return ax
end

function plot_inf_belief(hist::SimHist; figure::NamedTuple=(;), axis::NamedTuple=(;), kwargs...)
    fig = Figure(; figure...)
    ax = Axis(fig[1,1]; axis...)
    plot_inf_belief!(ax, hist; kwargs...)
    return fig
end

function _vectorize_inf(ps::Vector{ParticleCollection{S}}, pop::Int) where S <: CovidState
    T = length(ps)
    Np = n_particles(first(ps))
    x = Vector{Int}(undef, Np*T)
    y = Vector{Float64}(undef, Np*T)

    for (i,ci) in enumerate(CartesianIndices((Np,T)))
        p,t = Tuple(ci)
        x[i] = t
        s = ps[t].particles[p]
        y[i] = infected(s) / pop
    end
    return x, y
end

function _vectorize_inf_sample(ps::Vector{ParticleCollection{S}}, pop::Int, n_samples::Int) where S <: CovidState
    T = length(ps)
    x = Vector{Int}(undef, n_samples*T)
    y = Vector{Float64}(undef, n_samples*T)

    for (i,ci) in enumerate(CartesianIndices((n_samples,T)))
        p,t = Tuple(ci)
        x[i] = t
        s = rand(ps[t].particles)
        y[i] = infected(s) / pop
    end
    return x, y
end

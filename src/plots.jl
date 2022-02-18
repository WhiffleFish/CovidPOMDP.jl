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

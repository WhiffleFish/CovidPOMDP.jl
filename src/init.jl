"""
Fit Distributions to MC sim data for secondary infections per index case as a function of infection age

# Arguments
- `df::DataFrame` - DataFrame for csv containing MC simulations for daily individual infections.
- `horizon::Int=14` - Number of days in infection age before individual is considered naturally recovered and completely uninfectious.
- `sample_size::Int=50` - Sample size for `infections_path` csv where row entry is average infections for given sample size.
"""
function FitInfectionDistributions(df::DataFrame, horizon::Int=14, sample_size::Int=50)
    Λ = 1_000 # variance scaling parameter
    distributions = Vector{Gamma{Float64}}(undef, horizon)
    for day in 1:horizon
        try
            k, θ = Distributions.params(fit(Gamma, df[!,day]))
            distributions[day] = Gamma(k/(sample_size*Λ), Λ*θ*sample_size)
        catch e
            if e isa DomainError
                distributions[day] = Gamma(1e-100,1e-100)
            else
                throw(e)
            end
        end
    end
    return distributions
end


"""
Proportion of infectious population above some limit of detection from MC simulations

# Arguments
- `df::DataFrame` - DataFrame for csv containing MC simulations for daily individual infections.
- `day::Int` - Infection age (day)
- `LOD::Real` - Limit of detection (Log scale: ``10^x \\rightarrow x``)
"""
function prop_above_LOD(df::DataFrame, day::Int, LOD::Real)::Float64
    sum(df[!,day] .> LOD)/size(df,1)
end


"""
Return distribution resulting from sum of i.i.d RV's characterized by Gamma distribution

# Arguments
- `dist::Gamma` - Distribution characterizing random variable
- `N::Int` - Number of i.i.d RV's summed
"""
function RVsum(dist::Gamma, N::Int)
    k, θ = Distributions.params(dist)
    if k*N > 0
        return Gamma(k*N, θ)
    else
        return Gamma(1e-100, 1e-100)
    end
end

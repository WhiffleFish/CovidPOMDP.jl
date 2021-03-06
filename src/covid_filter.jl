function add_noise(p::InfParams, l = 0.10)
    inf = copy(p.infectiousness)
    p_test = copy(p.pos_test_probs)

    for (i,d) in enumerate(inf)
        k,θ = Distributions.params(d)
        k′ = k*(rand()*l+(1-l/2))
        θ′ = θ*(rand()*l+(1-l/2))
        inf[i] = Gamma(k′,θ′)

        λ = (k′*θ′) / (k*θ)
        p_test[i] *= λ
        p_test[i] = min(p_test[i], 1.0)
    end

    return InfParams(
        p_test,
        p.symptom_probs,
        p.asymptomatic_prob,
        p.symptomatic_isolation_prob,
        inf
    )
end

function param_noise(s::S, l) where S <: CovidState
    return S(statevars(s)..., add_noise(s.params, l), s.prev_action)
end

mutable struct NoisyCovidFilter{PM, P, PMEM, RNG<:AbstractRNG} <: Updater
    predict_model::PM
    particles::P
    weights::Vector{Float64}
    _particle_mem::PMEM
    _unique::Int
    rng::RNG
end

function NoisyCovidFilter(pomdp::POMDP{S}, n::Int) where S<:CovidState
    return NoisyCovidFilter(
        pomdp,
        ParticleCollection(Vector{S}(undef,n)),
        Vector{Float64}(undef, n),
        ParticleCollection(Vector{S}(undef,n)),
        n,
        Random.GLOBAL_RNG
    )
end

function noise_width(N::Int, Un::Int)
    return ((0.05 - 0.25)/(N-1))*(Un-N) + 0.05
end

function ParticleFilters.predict!(up::NoisyCovidFilter, b, a, o)
    pomdp = up.predict_model
    pm = up.particles.particles
    rng = up.rng

    N = length(pm)
    un = up._unique
    noise_param = noise_width(N, un)

    for (i,s) in enumerate(particles(b))
        sp, o, r = POMDPs.gen(pomdp, s, a, rng)
        pm[i] = param_noise(sp, noise_param)
    end
    return up.particles
end

function ParticleFilters.reweight!(up::NoisyCovidFilter, b, a, bp, o)
    pomdp = up.predict_model
    p1 = b.particles
    p2 = bp.particles
    wm = up.weights
    rng = up.rng

    w_sum = 0.0
    for i in eachindex(p1)
        s = p1[i]
        sp = p2[i]
        dist = POMDPs.observation(pomdp, s, a, sp)
        w = pdf(dist, o)
        w_sum += w
        wm[i] = w
    end

    return wm ./= w_sum # normalized
end

function ParticleFilters.resample(up::NoisyCovidFilter, p::Vector{S}, w::Vector{Float64}) where S
    N = length(p)
    rng = up.rng

    ps = Array{S}(undef, N)
    r = rand(rng)/N
    c = w[1]
    i = 1
    U = r
    for m in 1:N
        while U > c && i < N
            i += 1
            c += w[i]
        end
        U += inv(N)
        s = p[i]
        ps[m] = s
    end
    up._unique = length(unique(ps))
    return ParticleCollection(ps)
end

function ParticleFilters.update(up::NoisyCovidFilter, b::ParticleCollection, a::CovidAction, o::Int)
    b′ = predict!(up, b, a, o)
    w = reweight!(up, b, a, b′, o)
    return resample(up, b′.particles, w)
end

ParticleFilters.initialize_belief(pf::NoisyCovidFilter, b::ParticleCollection) = b

function ParticleFilters.initialize_belief(pf::NoisyCovidFilter, b)
    p = pf.particles.particles
    for i in eachindex(p)
        p[i] = rand(b)
    end
    return deepcopy(pf.particles)
end

function POMDPs.observation(pomdp::SingleCovidPOMDP, s::SingleCovidState, a::CovidAction, sp::SingleCovidState)
    iszero(a.testing_prop) && return CovidPOMDPs.UninformedDist() # not getting any info; any transition equally likely
    tot_mean = 0.0
    tot_variance = 0.0
    p = s.params
    for (i,inf) in enumerate(s.I)
        num_already_tested = sum(@view s.T[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,p.pos_test_probs[i])
        tot_mean += Statistics.mean(dist)
        tot_variance += Statistics.var(dist)
    end
    return Normal(tot_mean, sqrt(tot_variance))
end

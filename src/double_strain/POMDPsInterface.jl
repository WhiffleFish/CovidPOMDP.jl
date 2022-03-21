function POMDPs.observation(pomdp::DoubleCovidPOMDP, s::DoubleCovidState, a::CovidAction, sp::DoubleCovidState)
    iszero(a.testing_prop) && return CovidPOMDPs.UninformedDist() # not getting any info; any transition equally likely
    tot_mean = 0.0
    tot_variance = 0.0
    p1 = pomdp.params
    p2 = s.params
    for (i,inf) in enumerate(s.I1)
        num_already_tested = sum(@view s.T1[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,p1.pos_test_probs[i])
        tot_mean += Statistics.mean(dist)
        tot_variance += Statistics.var(dist)
    end

    for (i,inf) in enumerate(s.I2)
        num_already_tested = sum(@view s.T2[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        dist = Binomial(num_tested,p2.pos_test_probs[i])
        tot_mean += Statistics.mean(dist)
        tot_variance += Statistics.var(dist)
    end

    return Normal(tot_mean, sqrt(tot_variance))
end

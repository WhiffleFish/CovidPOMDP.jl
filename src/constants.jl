const INFECTION_HORIZON = 14

const INFECTION_DATA_PATH = joinpath(@__DIR__, "data", "Sample50.csv")

const INF_DATAFRAME = INFECTION_DATA_PATH |> File |> DataFrame

const INF_DIST = FitInfectionDistributions(INF_DATAFRAME, INFECTION_HORIZON, 50)

const VIRAL_LOADS_DATA_PATH = joinpath(@__DIR__, "data", "raw_viral_load.csv")

const VIRAL_LOADS = VIRAL_LOADS_DATA_PATH |> File |> DataFrame

const LIMIT_OF_DETECTION = 6

const POS_TEST_PROBS = [
    prop_above_LOD(VIRAL_LOADS, day, LIMIT_OF_DETECTION)
    for day in 1:INFECTION_HORIZON
]

const SYMPTOMATIC_ISOLATION_PROB = 0.30

const ASYMPTOMATIC_PROB = 0.10

const SYMPTOM_DIST = LogNormal(1.644, 0.363)

const SYMPTOM_PROBS = [
    cdf(SYMPTOM_DIST, i) - cdf(SYMPTOM_DIST, i-1) for i in 1:INFECTION_HORIZON
]

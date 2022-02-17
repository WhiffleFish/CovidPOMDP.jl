const INFECTION_DATA_PATH = joinpath(@__DIR__, "data", "Sample50.csv")

const INF_DATAFRAME = File(INFECTION_DATA_PATH) |> DataFrame

const INF_DIST = FitInfectionDistributions(df, horizon, sample_size)

const VIRAL_LOADS_DATA_PATH = joinpath(@__DIR__, "data", "raw_viral_load.csv")

const VIRAL_LOADS = File(viral_loads_path) |> DataFrame

const LIMIT_OF_DETECTION = 6

const INFECTION_HORIZON = 14

const POS_TEST_PROBS = [
    prop_above_LOD(VIRAL_LOADS, day, LIMIT_OF_DETECTION)
    for day in 1:INFECTION_HORIZON
]

const SYMPTOM_DIST = LogNormal(1.644, 0.363)

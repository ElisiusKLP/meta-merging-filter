
#loading packages
using Distributions
using ActionModels, HierarchicalGaussianFiltering
using Turing

"""
---------------------------
"""

# loading in "original_action_model"
include("/work/HGF/bachelor_repo/action_functions/bci_action.jl")

original_params = Dict(
    #PArametrs: 
    "p_common" => 0.5 ,# p_common - prior for Common or not
    "muP" => 0,# muP - centrality bias
    "sigP" => 1,# sigP - position variability from 0
    "sigA" => 1,# sigA - auditory noise
    "sigV" => 1,# sigV - visual noise
    "action_noise" => 1
)

#States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position

original_states = [
    Dict("name" => "C"),
    Dict("name" => "sAV"),
    Dict("name" => "sA"),
    Dict("name" => "sV"),
]

original_states = Dict(
    "C" => 0.5,
    "sAV" => 0,
    "sA" => 0,
    "sV" => 0,
)

priors = Dict(
    "p_common" => Beta(1,1),
    "sigP" => Uniform(0,30),
    "sigA" => Uniform(0,30),
    "sigV" => Uniform(0,30),
    "action_noise" => Uniform(0,30),
)


agent = init_agent(
    original_action_model,
    parameters = original_params,
    states = original_states
)

get_parameters(agent)

"""
2. load sim data
"""
# load simulation 1 data

using CSV
using DataFrames

file_path = "/work/HGF/simulation/sim_data/data_sim1_2024-09-14_174512.csv"

# Load the CSV file into a DataFrame
df = CSV.read(file_path, DataFrame)


# Convert all column names to lowercase
rename!(df, Dict(col => lowercase(col) for col in names(df)))

println("First few rows of the DataFrame:")
println(first(df, 5))

input_vector = Vector{Tuple{Float64, Float64}}()

# Create the vector using comprehension
input_vector = [(row.auditory_position, row.visual_position) for row in eachrow(df)]

""" 3. give input data to agent """

give_inputs!(agent, inputs)
get_history(agent)
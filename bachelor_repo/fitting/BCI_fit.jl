"""
Try to run BCI mode_fit
"""

# set wd to subfolder in ucloud
cd("/work/Multisensory_HGF")

# Packages
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays
using Distributed # for parallelization

workers()

# reset workers if they don't want to initialize
if length(workers()) > 1
    worker_ids = workers()
    rmprocs(worker_ids)
end
# Setup workers
n_cores = 25
if n_cores > 1
    addprocs(n_cores, exeflags = "--project")
    @everywhere @eval using HierarchicalGaussianFiltering
    @everywhere @eval using ActionModels
    
end


"""
Creating the BCI in action models framework
without using a HGF substruct 
"""

#Common, independent
# prior for common: P_common

#Hvis common: location S_AV
# Gaussian(mean_AV, sigP) 


#Parameters: 
# p_common - prior for Common or not
# muP - centrality bias
# sigP - position variability from 0
# sigA - auditory noise
# sigV - visual noise

#States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position

#observations
# xA
# xV

# loading in "original_action_model"
@everywhere include("$(pwd())/action_functions/bci_action.jl")

# If i wanted to save the posterior_C 
#agent.states["C"] = posterior_C
#push!(agent.history["C"], posterior_C)

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
# prøv at fit uden muP
# og smallere priors


agent = init_agent(
    original_action_model,
    parameters = original_params,
    states = original_states
)

get_parameters(agent)

# SIMULATING DATA
values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
actions = Array(rand(values, num_samples))
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = Array( [rand(values, 2) for _ in 1:num_samples] )
# To view the first few samples
println(inputs[1:10])

get_parameters(agent)

give_inputs!(agent, inputs)
get_history(agent)
plot_trajectory(agent, "C") 
plot_trajectory(agent, "action")

inputs

# FITTING REAL DATA

dataset = CSV.read("$(pwd())/dataset/park_and_kayser2023.csv", DataFrame)

df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]

# Fitting independent group models

input_cols = [:auditory_location, :visual_location]
action_cols = [:action]
independent_group_cols = [:subject]

results = fit_model(
    agent,
    priors,
    df_exp1;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    n_iterations = 10000,
    n_cores = 25,
    n_chains = 2,
)

serialize("bci_fit10_exp1_23-12-23.jls", chains)

# CREATE TURING MODEL
include("$(pwd())/src/create_turing_obj.jl")
include("$(pwd())/src/prefit_checks.jl")
using Logging
include("$(pwd())/src/fitting_helper_functions.jl")
include("$(pwd())/src/structs.jl")

get_parameters(agent)

model = create_turing_obj(
    agent,
    priors,
    df_exp1;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    n_iterations = 10000,
    n_cores = 25,
    n_chains = 2,
)

model[1]
#fit9 = deserialize("/work/Multisensory_HGF/chain_saved/bci_fit4_exp1_23-12-23.jls")

prior = Turing.sample(model[1], Prior(), 1000)

## Inspecting results from experiment 1
xy = chains[String31("participant 1.13")]

plot(xy)

plot_parameter_distribution(xy, priors)

dist = rand(LogNormal(0,3),1000)
dist[dist .< 200] .= 200

histogram(dist, xlims = [0,100])

dist = rand(truncated(Normal(5,1),0,30),1000)
dist[dist .< 200] .= 200

histogram(dist, xlims = [0,100])

# Fitting the model to the simulated data
chains = fit_model(
    agent,
    priors,
    inputs_exp1,
    actions_exp1,
    n_iterations = 2000,
)

write("bci_fit2_21-12-23.jls", chains)

part = chains[String15("experiment 1"), String31("participant 1.15")]

plot(part)

plot_parameter_distribution(part, priors)

plot_predictive_simulation(
    part,
    agent,
    inputs_exp1,
    ("action");
    n_simulations = 3
)

get_posteriors(part)

#----------------
##NOTES
#Fitting group level models

input_cols = [:auditory_location, :visual_location]
action_cols = [:action]
independent_group_cols = [:experiment, :participant]

chains = fit_model(
    agent,
    priors,
    dataset;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    n_iterations = 1000,
    n_cores = 4,
    n_chains = 2,
)






input_cols = [:auditory_input, :visual_input]
action_cols = [:action]
independent_group_cols = [:experiment]
multilevel_cols = [:participant]

priors = Dict(
    ("xA", "input_noise") => Multilevel(
        :participant,
        LogNormal,
        ["xA_noise_group_mean", "xA_noise_group_sd"]
    ),
    "xA_noise_group_mean" => Normal(0, 1),
    "xA_noise_group_sd" => LogNormal(0, 1),
)

chains = fit_model(
    agent,
    priors,
    dataset;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    multilevel_cols = multilevel_cols,
    n_iterations = 1000,
    n_cores = 4,
    n_chains = 2,
)



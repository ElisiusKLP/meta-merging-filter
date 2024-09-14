"""
Creating Forced-Fusion model with HGF.
Feeding with random (no patterns) simulated data/locations.

Trying to conduct parameter recovery.

"""
# Packages
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays


#List of input nodes to create
input_nodes = [Dict(

    "name" => "A",
), Dict("name" => "V",)]

#List of state nodes to create
state_nodes = [
    Dict(
        "name" => "location",
    ),
]
#List of child-parent relations
edges = [
    Dict(
        "child" => "A",
        "value_parents" => ("location"),
    ),
    Dict(
        "child" => "V",
        "value_parents" => ("location"),
    ),
]
#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
)

get_parameters(hgf)

function ff_multisensory_hgf_action(agent::Agent, input)
    action_noise = agent.parameters["action_noise"]
    #Update hgf
    hgf = agent.substruct
    update_hgf!(hgf, input)
    #get out inferred location
    inferred_location = get_states(hgf, ("location", "posterior_mean"))
    #Create action distribution
    action_distribution = Normal(inferred_location, action_noise)
    return action_distribution
end

agent_parameters = Dict(
    "action_noise" => 1
)

agent = init_agent(
    ff_multisensory_hgf_action,
    parameters = agent_parameters,
    substruct = hgf,
)

priors = Dict(
    ("V", "input_noise") => Normal(-2, 1),
    ("A", "input_noise") => Normal(-2, 1),
    "action_noise" => LogNormal(0,0.2),
)


x = LogNormal(0,0.2)
plot(x)

get_parameters(agent)
# Creating a simulation of five spatial locations similar to ventriliquist experiments
## Define the discrete values

values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
actions = rand(values, num_samples)
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = [rand(values, 2) for _ in 1:num_samples]
# To view the first few samples
println(inputs[1:10])

# Feeding the agent with inputs to generate some actions

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "A")
plot_trajectory!(agent, "V")
plot_trajectory!(agent, "location")
plot_trajectory!(agent, "action")


action_history = get_history(agent, "action")

action_history = action_history[2:1001]

# FITTING MODEL from scratch
## Here I use the actions simulated from the agent model to fit the model
results = fit_model(
    agent,
    priors,
    inputs,
    action_history
)


plot(results)

get_posteriors(results)

CSV.write("", results) ## write csv

serialize("forces-fusion-26-10-23.jls", results) # saveing the mcmc chains object (model)

results = deserialize("forces-fusion-26-10-23.jls") # loading in serialized MCMCChain object

typeof(results)

plot(results, ("A","evolution_rate") )

# plot trajectory of agent locations and actions
reset!(agent)
give_inputs!(agent, inputs)
plot_trajectory(agent, "location")
plot_trajectory!(agent, "action")

histogram(inputs)
histogram(actions)

# the prior predictive simulation tells us the distribution of observed data we expect before we have observed any data

results

plot_parameter_distribution(results, priors)

plot_parameter_distribution(results)

plot_predictive_simulation(
    priors,
    agent,
    inputs,
    ("location", "posterior_mean");
    n_simulations = 3
)

plot_trajectory!(agent, "A")

plot_predictive_simulation(
    results,
    agent,
    inputs,
    ("action");
    n_simulations = 3
)

plot_trajectory!(agent, "action")

give_inputs!(agent, inputs)
reset!(agent)

plot_trajectory(agent, "action")
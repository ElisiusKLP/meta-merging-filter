"""
Creating Segregated model with HGF.
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
input_nodes = [
    Dict(
    "name" => "X_A",
), Dict("name" => "X_V",)]

#List of state nodes to create
state_nodes = [
    Dict(
        "name" => "S_A",
    ), Dict(
        "name" => "S_V",
    )
]

#List of child-parent relations
edges = [
    Dict(
        "child" => "X_A",
        "value_parents" => ("S_A"),
    ),
    Dict(
        "child" => "X_V",
        "value_parents" => ("S_V"),
    ),
]
#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
)

get_parameters(hgf)

function seg_multisensory_hgf_action(agent::Agent, input, constant_cue = "A")
    """
    The segregated function takes an input of a three length vector of the form [auditory_stimulus, visual_stimulus, cue] 
    if the cue is not specified it defaults to "A" for auditory cue, but can be changed to visual. 
    """

    if length(input) == 3
        auditory_stimulus = input[1]
        visual_stimulus = input[2]
        cue = input[3]
    else
        auditory_stimulus = input[1]
        visual_stimulus = input[2]
        cue = constant_cue
    end
    
    action_noise = agent.parameters["action_noise"]
    #Update hgf with locations to infer from 
    hgf = agent.substruct

    update_hgf!(hgf, [auditory_stimulus, visual_stimulus])

    if cue == "A"

        inferred_position = get_states(hgf, ("S_A", "posterior_mean"))

    elseif cue == "V"

        inferred_position = get_states(hgf, ("S_V", "posterior_mean"))

    end
    
    #Create action distribution 
    action_distribution = Normal(inferred_position, action_noise)

    return action_distribution
end

agent_parameters = Dict(
    "action_noise" => 1
)

agent = init_agent(
    seg_multisensory_hgf_action,
    parameters = agent_parameters,
    substruct = hgf,
)

priors = Dict(
    ("X_V", "input_noise") => Normal(-2, 1),
    ("X_A", "input_noise") => Normal(-2, 1),
)


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
print(get_parameters(agent))
set_parameters!(
    agent,
    Dict(
        ("V", "evolution_rate") => -5,
        ("A", "evolution_rate") => 2,
    )
)
get_parameters(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "X_A")
plot_trajectory!(agent, "X_V")
plot_trajectory!(agent, "S_A")
plot_trajectory!(agent, "S_V")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history = action_history[2:1001]

init_params = get_parameters(agent)

series


# FITTING MODEL from scratch


results = fit_model(
    agent,
    priors,
    inputs,
    action_history
)


plot(results)

x = get_posteriors(results, type = "median")
x["V", "evolution_rate"]

get_parameters(hgf)
get_parameters(agent)
ActionModels.get_parameters(agent)

# the prior predictive simulation tells us the distribution of observed data we expect before we have observed any data

results

plot_parameter_distribution(results, priors)

plot_parameter_distribution(results)

plot_predictive_simulation(
    priors,
    agent,
    inputs,
    ("S_A", "posterior_mean");
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

#loading packages
using Distributions
using ActionModels, HierarchicalGaussianFiltering
using Turing


"This is for the merging HGF"
nodes = [
    # SEGREGATION
	ContinuousInput(
		name = "Seg_uA",
		input_noise = -2,
		bias = 0
	),
	ContinuousState(
		name = "Seg_sA",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "Seg_uV",
		input_noise = -2,
		bias = 0
	),
	ContinuousState(
		name = "Seg_sV",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
    # FORCED FUSION
	ContinuousState(
		name = "FF_sAV",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "FF_uA",
		input_noise = -2,
		bias = 0
	),
	ContinuousInput(
		name = "FF_uV",
		input_noise = -2,
		bias = 0
	),
    # META HGF
    ContinuousState(
		name = "Meta_sCOM",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "Meta_uCOM",
		input_noise = -2,
		bias = 0
	),
    ContinuousState(
		name = "Meta_sIND",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "Meta_uIND",
		input_noise = -2,
		bias = 0
	),
    #=
	ContinuousState(
		name = "Meta_vopCOM",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	)
        =#
]

edges = Dict(
	("Seg_uA", "Seg_sA") => ObservationCoupling(),
	("Seg_uV", "Seg_sV") => ObservationCoupling(),
	("FF_uA", "FF_sAV") => ObservationCoupling(),
	("FF_uV", "FF_sAV") => ObservationCoupling(),
	("Meta_uCOM", "Meta_sCOM") => ObservationCoupling(),
	("Meta_uIND", "Meta_sIND") => ObservationCoupling(),
)

#CHANGE THIS TO THE CORRECT ORDER
update_order = ["Seg_uA", "Seg_sA", "Seg_uV", "Seg_sV", "FF_uA", "FF_uV", "FF_sAV", "Meta_uCOM", "Meta_sCOM", "Meta_uIND", "Meta_sIND"]


#=
shared_parameters = Dict(
    "sAV_initial_mean" => (0, [("Seg_sA", "initial_mean"),("Seg_sV", "initial_mean"), ("FF_sAV", "initial_mean")]),
    "sAV_drift" => (0, [("Seg_sA", "drift"),("Seg_sV", "drift"), ("FF_sAV", "drift")]),
    "A_input_noise" => (0, [("FF_A", "input_noise"), ("Seg_A", "input_noise")]),
    "V_input_noise" => (0, [("FF_V", "input_noise"), ("Seg_V", "input_noise")]),
    "sAV_initial_precision" => (0, [("Seg_sA", "initial_precision"),("Seg_sV", "initial_precision"), ("FF_sAV", "initial_precision")]),
)
    =#

# fix init_precision to 100

hgf = init_hgf(
    nodes = nodes,
    edges = edges,
    update_order = update_order
)


print(get_parameters(hgf))
# load in "merging_hgf_action.jl"
pwd()
include("$(pwd())/action_functions/meta_merging_hgf_action.jl")

print(get_states(hgf))

# Create agent structure

agent_parameters = Dict(
    "action_noise" => 1,
)

agent = init_agent(
    meta_merging_hgf_action,
    parameters = agent_parameters,
    substruct = hgf,
)

get_parameters(agent)

"""
2. load sim data
"""
# load simulation 1 data

using CSV
using DataFrames

file_path = "/work/HGF/simulation/sim_data/data_sim2_2024-09-19_133138.csv"

# Load the CSV file into a DataFrame
df = CSV.read(file_path, DataFrame)


# Convert all column names to lowercase
rename!(df, Dict(col => lowercase(col) for col in names(df)))

println("First few rows of the DataFrame:")
println(first(df, 5))

input_vector =[]
# Create the vector using comprehension
input_vector = [ [row.auditory_position, row.visual_position, row.auditory_position, row.visual_position] for row in eachrow(df) ]

""" 3. Give Inputs """

reset!(agent)
give_inputs!(agent, input_vector)
agent_history = get_history(agent)

hgf.ordered_nodes.input_nodes[6]

action_history = agent_history["action"][2:end]

agent_history

# Step 1: Determine the maximum vector length
vector_lengths = map(length, values(agent_history))
maxlen = maximum(vector_lengths)

# Step 2: Initialize an empty DataFrame
df = DataFrame()

# Step 3: Iterate over the dictionary and populate the DataFrame
for (key, vec) in agent_history
    # Generate a column name by joining the tuple elements
    colname = string(key[1], "_", key[2])
    
    # Pad the vector with `missing` values if necessary
    padded_vec = if length(vec) < maxlen
        vcat(vec, fill(missing, maxlen - length(vec)))
    else
        vec[1:maxlen]  # Truncate if necessary
    end
    
    # Add the padded vector as a new column in the DataFrame
    df[!, Symbol(colname)] = padded_vec
end

# Display the resulting DataFrame
show(first(df, 5))

# Get summary statistics
stats = describe(df)

# Display the statistics
show(stats, allrows=true, allcols=true)

""" OLDn """
plot_trajectory(hgf, "FF_sAV")
action_history = agent_history["action"][2:end]

agent_parameters = Dict(
    "action_noise" => 1,
    "p_common" => 0.5
)

agent = init_agent(
    merging_hgf_w_reset,
    parameters = agent_parameters,
    substruct = hgf,
)

get_parameters(agent)

priors = Dict(
    ("FF_A", "input_noise") => Normal(-2, 1),
    ("FF_V", "input_noise") => Normal(-2, 1),
    ("Seg_A", "input_noise") => Normal(-2, 1),
    ("Seg_V", "input_noise") => Normal(-2, 1),
    "p_common" => Beta(1,1),
    "action_noise" => LogNormal(0,1),
)


# SIMULATION
# Inputs

values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
actions = Array(rand(values, num_samples))
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = [[v..., v...] for v in [rand(values, 2) for _ in 1:num_samples]]
# To view the first few samples
println(inputs[1:10])

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "FF_A")
plot_trajectory!(agent, "FF_V")
plot_trajectory!(agent, "Seg_A")
plot_trajectory!(agent, "Seg_V")
plot_trajectory!(agent, "Ind_sV")
plot_trajectory!(agent, "C")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history

# fit simulated data

# FIT EXP 1

dataset = CSV.read("$(pwd())/dataset/park_and_kayser2023.csv", DataFrame)

dataset[!, :auditory_location2] = dataset[!, :auditory_location]
dataset[!, :visual_location2] = dataset[!, :visual_location]

df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]

# Fitting independent group models

# i have to copy location columns as fit_model need unique arrays

input_cols = [:auditory_location, :visual_location, :auditory_location2, :visual_location2]
action_cols = [:action]
independent_group_cols = [:subject]

results = fit_model(
    agent,
    priors,
    df_exp1;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    multilevel_group_cols
    n_iterations = 10000,
    n_cores = 25,
    n_chains = 2,
)

serialize("merging_fit_exp1_23-12-23.jls", results)

plot(results)

# TRYING WITH DATASET

# load in dataset
dataset = CSV.read("park_and_kayser2023.csv", DataFrame)

show(dataset)

#doin some cleaning to get the numerical values
dataset[!, "input"] = JSON.parse.(dataset[!, "input"])

typeof(dataset[!,"input"])

typeof(dataset[!,"action"])

show(dataset)

inputs = dataset[:,1]

inputs

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "xA")
plot_trajectory!(agent, "xV")
plot_trajectory!(agent, "FF_sAV")
plot_trajectory!(agent, "Ind_sA")
plot_trajectory!(agent, "Ind_sV")
plot_trajectory!(agent, "C")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history

# Fitting model with all experiments multilevel

# I try with experiment 1 only
df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]
inputs_exp1 = df_exp1[:,1]
actions_exp1 = df_exp1[:,2]

actions_exp1

reset!(agent)
give_inputs!(agent, inputs_exp1)

plot_trajectory(agent, "xA")
plot_trajectory!(agent, "xV")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")
action_history_exp1 = action_history[2:length(action_history)]
action_history_exp1 = ForwardDiff.value.(action_history_exp1)


action_history_exp1
print(action_history_exp1)

# save csv of inputs_exp1 and actions_history_exp1
df_exp1_simulated_actions = DataFrame(
    input = inputs_exp1,
    action = action_history_exp1,
)

df_exp1_simulated_actions

CSV.write("df_exp1_simulated_actions.csv", df_exp1_simulated_actions)

chains = fit_model(
    agent,
    priors,
    inputs_exp1,
    actions_exp1,
    n_iterations = 2000,
)


include("$(pwd())/src/create_turing_obj.jl")

create
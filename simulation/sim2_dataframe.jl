using Random, Distributions, DataFrames, CSV, Dates
include("simulator_2.jl")

# Function to run simulation 2 and save data
function run_simulation_2_and_save(
    n_timesteps::Int, 
    prob_common::Float64, 
    noise_std::Float64, 
    signal_noise_std::Float64, 
    azimuth_min::Float64, 
    azimuth_max::Float64,
    initial_bound_std::Float64,
    boundary_timesteps::Vector{Int},
    boundary_std_values::Vector{Float64},
    starting_position::String = "shared"
)
    # Run the simulation from simulator_2 module
    signal_data, boundary_std_data = simulator_2.simulate_experiment_with_discrete_boundary(
        n_timesteps, prob_common, noise_std, signal_noise_std, 
        azimuth_min, azimuth_max, initial_bound_std, boundary_timesteps, boundary_std_values, starting_position)
    
    # Create vectors to hold the dataframe columns
    timesteps = Vector{Int}(undef, n_timesteps)
    auditory_positions = Vector{Float64}(undef, n_timesteps)
    visual_positions = Vector{Float64}(undef, n_timesteps)
    causal_structure = Vector{Bool}(undef, n_timesteps)
    boundary_std = Vector{Float64}(undef, n_timesteps)
    common_source_positions = Vector{Float64}(undef, n_timesteps)
    
    # Populate vectors with simulation data
    for t in 1:n_timesteps
        timesteps[t] = t
        auditory_positions[t] = signal_data[t][1][1]  # Auditory position
        visual_positions[t] = signal_data[t][1][2]    # Visual position
        causal_structure[t] = signal_data[t][2]       # Causal structure (common or independent source)
        boundary_std[t] = boundary_std_data[t][1]     # Boundary standard deviation at time t
        common_source_positions[t] = boundary_std_data[t][2]  # Common source position at time t
    end
    
    # Create a dataframe
    df = DataFrame(
        Timestep = timesteps,
        Auditory_Position = auditory_positions,
        Visual_Position = visual_positions,
        Causal_Structure = causal_structure,
        Boundary_Std = boundary_std,
        Common_Source_Position = common_source_positions,
        Prob_Common = fill(prob_common, n_timesteps),
        Noise_Std = fill(noise_std, n_timesteps),
        Signal_Noise_Std = fill(signal_noise_std, n_timesteps),
        Initial_Bound_Std = fill(initial_bound_std, n_timesteps),
        Azimuth_Min = fill(azimuth_min, n_timesteps),
        Azimuth_Max = fill(azimuth_max, n_timesteps),
        Starting_Position = fill(starting_position, n_timesteps)
    )
    
    # Generate a filename with the current date and time
    current_datetime = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    save_path = "/work/HGF/simulation/sim_data/data_sim2_$(current_datetime).csv"
    
    # Save dataframe as a CSV file
    CSV.write(save_path, df)
    println("Data saved to: $save_path")
end

# Set parameters and run the simulation
n_timesteps = 10000
prob_common = 0.3
noise_std = 2.0
signal_noise_std = 0.5
azimuth_min = -30.0
azimuth_max = 30.0
initial_bound_std = 1.0  # Initial boundary standard deviation

# Define boundary timesteps and corresponding standard deviations
boundary_timesteps = [2000, 4000, 6000, 8000]
boundary_std_values = [4.0, 2.0, 6.0, 1.0]

starting_position = "shared"  # or "random"

# Run the simulation and save the data
run_simulation_2_and_save(
    n_timesteps, prob_common, noise_std, signal_noise_std, azimuth_min, azimuth_max, 
    initial_bound_std, boundary_timesteps, boundary_std_values, starting_position)

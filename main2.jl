module Simulator_2

using Random, Distributions

# Define a struct for the sources
struct Source
    position::Float64
    signal_noise::Normal
end

# Function to update the source position with Gaussian noise (random walk)
function move_source(source::Source, noise_std::Float64, azimuth_min::Float64, azimuth_max::Float64)
    movement_noise = rand(Normal(0, noise_std))
    new_position = source.position + movement_noise

    # Reflective boundary conditions
    if new_position < azimuth_min
        new_position = azimuth_min + (azimuth_min - new_position)
    elseif new_position > azimuth_max
        new_position = azimuth_max - (new_position - azimuth_max)
    end

    return Source(new_position, source.signal_noise)
end

# Function to sample a signal with noise around the source position
function sample_signal(source::Source)
    signal_noise = rand(source.signal_noise)
    return source.position + signal_noise
end

# Function to create Gaussian bound around the common source
function gaussian_bound(center::Float64, bound_std::Float64, azimuth_min::Float64, azimuth_max::Float64)
    bound_center = rand(Normal(center, bound_std))
    return clamp(bound_center, azimuth_min, azimuth_max)
end

# Function to simulate the experiment over n timesteps with dynamic Gaussian bound
function simulate_experiment_with_dynamic_bound(
    n_timesteps::Int, 
    prob_common::Float64, 
    noise_std::Float64, 
    signal_noise_std::Float64, 
    azimuth_min::Float64, 
    azimuth_max::Float64,
    initial_bound_std::Float64,
    growth_timesteps::Vector{Int},
    bound_stds::Vector{Float64},
    starting_position::String = "shared"
)
    # Initialize starting positions based on settings
    if starting_position == "shared"
        shared_starting_position = Source(rand(azimuth_min:azimuth_max), Normal(0, signal_noise_std))
        common_source = shared_starting_position
        independent_light = shared_starting_position
        independent_sound = shared_starting_position
    elseif starting_position == "random"
        common_source = Source(rand(azimuth_min:azimuth_max), Normal(0, signal_noise_std))
        independent_light = Source(rand(azimuth_min:azimuth_max), Normal(0, signal_noise_std))
        independent_sound = Source(rand(azimuth_min:azimuth_max), Normal(0, signal_noise_std))
    end

    # Initialize an empty vector to store results
    signal_data = Vector{Tuple{Vector{Float64}, Bool}}(undef, n_timesteps)
    boundary_stds = []

    for t in 1:n_timesteps
        # Compute the current boundary std based on discrete growth
        if t in growth_timesteps
            idx = findfirst(isequal(t), growth_timesteps)
            if idx !== nothing
                current_bound_std = bound_stds[idx]
            end
        else
            current_bound_std = initial_bound_std
        end
        boundary_stds = push!(boundary_stds, current_bound_std)

        # Decide whether to use the common source based on probability
        is_common = rand() < prob_common

        # Update the source positions
        common_source = move_source(common_source, noise_std, azimuth_min, azimuth_max)
        bound_center = gaussian_bound(common_source.position, current_bound_std, azimuth_min, azimuth_max)
        independent_light = move_source(Source(bound_center, Normal(0, signal_noise_std)), noise_std, azimuth_min, azimuth_max)
        independent_sound = move_source(Source(bound_center, Normal(0, signal_noise_std)), noise_std, azimuth_min, azimuth_max)

        if is_common
            # Sample signals from the common source
            auditory_signal = sample_signal(common_source)
            visual_signal = sample_signal(common_source)
        else
            # Sample signals from the independent sources
            auditory_signal =

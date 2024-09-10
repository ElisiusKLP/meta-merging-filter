module Simulator

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

# Function to simulate the experiment over n timesteps
function simulate_experiment(
    n_timesteps::Int, 
    prob_common::Float64, 
    noise_std::Float64, 
    signal_noise_std::Float64, 
    azimuth_min::Float64, 
    azimuth_max::Float64,
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

    for t in 1:n_timesteps
        # Decide whether to use the common source based on probability
        is_common = rand() < prob_common
        
        common_source = move_source(common_source, noise_std, azimuth_min, azimuth_max)
        independent_light = move_source(independent_light, noise_std, azimuth_min, azimuth_max)
        independent_sound = move_source(independent_sound, noise_std, azimuth_min, azimuth_max)

        if is_common
            auditory_signal = sample_signal(common_source)
            visual_signal = sample_signal(common_source)
        else
            auditory_signal = sample_signal(independent_sound)
            visual_signal = sample_signal(independent_light)
        end

        # Store the auditory and visual signals along with the flag for common source
        signal_data[t] = ([auditory_signal, visual_signal], is_common)
    end

    return signal_data
end

end

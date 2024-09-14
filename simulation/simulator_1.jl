"""

This simulation models a scenario where auditory and visual stimuli are 
emitted from sources moving along a predefined azimuth range, typically 
set between -22 and +22 degrees. Each time step simulates the positions 
of three source points along this azimuth: one common source emitting 
both auditory and visual signals, and two independent sources emitting 
only one signal type each.

## Key Parameters:
- **Azimuth Range**: Defines the angular space for the sources, 
  bounded by a minimum (e.g., -22 degrees) and maximum (e.g., +22 degrees) value.
- **Number of Timesteps**: Specifies how many steps the simulation will run.
- **Common Source Probability**: Predefined probability of signals 
  coming from the common source at each timestep.
- **Source Movement**: Each source follows a Gaussian random walk, 
  updating its position with added Gaussian noise at each timestep.
- **Gaussian Noise**: The common source emits auditory and visual signals 
  with Gaussian noise around the source's true position, simulating 
  imprecision in the signals.

## Simulation Steps:
1. **Initialization**: Three sources are randomly initialized within 
   the azimuth range, with one common source and two independent sources.
2. **Timestep Simulation**: 
   - For each timestep, a random decision determines whether the signals 
     are emitted from the common or independent sources.
   - Each source's position is updated using a Gaussian random walk, 
     with reflective boundaries at the azimuth limits.
   - The positions of auditory and visual signals are recorded for each step.
3. **Recording Data**: The auditory and visual signal positions are stored, 
   along with whether the signals were sourced from a common or independent 
   source.
4. **Output**: The simulation generates a time series of auditory and 
   visual signal positions for analysis or visualization.
"""

module simulator_1

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

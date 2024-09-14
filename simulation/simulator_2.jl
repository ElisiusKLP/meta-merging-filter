"""
# Second Simulation Description

The second simulation extends the basic model by introducing **dynamic 
Gaussian boundaries** around the common source. This fluctuation affects 
only the **deviation from the common cause source** and **not the signal noise**. 
This dynamic is artificially introduced to control the induction of the 
ventriloquist effect in the participant. 

## Key Concept: Noise
- **Signal Noise**: Defines the reliability of each signal (auditory and visual), 
  representing the inherent uncertainty in the emitted signals.
- **Walk Noise**: Analogous to the noise in the underlying physical movement 
  of the source in space, this affects how the source position is updated.

## Key Differences:
- **Dynamic Boundary Growth**: The Gaussian boundary of the common source 
  varies according to predefined arrays of timesteps and standard deviations. 
  This fluctuation controls the deviation of the signals from the common 
  cause source and simulates the induction of the ventriloquist effect.
- **Effect on Independent Sources**: The independent sources are not directly 
  constrained by this boundary but continue to emit their signals with 
  standard signal and walk noise.
- **Shared Boundary**: Both auditory and visual signals emitted from the 
  common source are influenced by the same dynamic boundary, affecting their 
  deviation from the common source position.

## Simulation Steps:
1. **Initialization**: Similar to the basic simulation, but with a dynamic 
   Gaussian boundary starting from an initial standard deviation.
2. **Timestep Simulation**: 
   - The common source probability still determines whether signals 
     are from the common or independent sources.
   - The common source boundary changes dynamically over time, controlling 
     the deviation of auditory and visual signals from the common cause 
     source, while the inherent signal noise remains constant.
3. **Recording Data**: As in the basic simulation, but with the addition 
   of recording the current boundary’s standard deviation and its effect 
   on signal deviation from the common source.
4. **Output**: The output contains the time series of signal positions, 
   with the dynamic boundary reflecting the changing signal deviation.
"""

module simulator_2

using Random, Distributions

"""
Simulator 2: Controlled Discrete Gaussian Boundary Simulation
"""

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

# Function to simulate the experiment over n timesteps with discrete boundary growth
function simulate_experiment_with_discrete_boundary(
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
    boundary_std_data = Vector{Tuple{Float64, Float64}}()

    
    # Initialize the boundary standard deviation
    current_bound_std = initial_bound_std

    for t in 1:n_timesteps
        # Check if the current timestep is in the boundary_timesteps array
        if t in boundary_timesteps
            # Find the index of the current timestep in the boundary_timesteps array
            index = findfirst(x -> x == t, boundary_timesteps)
            # Update the boundary standard deviation
            current_bound_std = boundary_std_values[index]
        end

        # Decide whether to use the common source based on probability
        is_common = rand() < prob_common

        # Update the source positions
        common_source = move_source(common_source, noise_std, azimuth_min, azimuth_max)
        bound_center_light = gaussian_bound(common_source.position, current_bound_std, azimuth_min, azimuth_max)
        bound_center_sound = gaussian_bound(common_source.position, current_bound_std, azimuth_min, azimuth_max)
        independent_light = move_source(Source(bound_center_light, Normal(0, signal_noise_std)), noise_std, azimuth_min, azimuth_max)
        independent_sound = move_source(Source(bound_center_sound, Normal(0, signal_noise_std)), noise_std, azimuth_min, azimuth_max)

        
        # Store the current boundary std
        current_common_source_position = common_source.position
        push!(boundary_std_data, (current_bound_std, current_common_source_position))

        # sampling process
        if is_common
            # Sample signals from the common source
            auditory_signal = sample_signal(common_source)
            visual_signal = sample_signal(common_source)
        else
            # Sample signals from the independent sources
            auditory_signal = sample_signal(independent_sound)
            visual_signal = sample_signal(independent_light)
        end

        # Store the auditory and visual signals along with the flag for common source
        signal_data[t] = ([auditory_signal, visual_signal], is_common)
    end

    return signal_data, boundary_std_data
end

end

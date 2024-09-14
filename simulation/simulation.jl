"""

"""

# Installation
## -> Pkg add DataFrames, CSV, Distributions, Plots, ActionModels, HierarchicalGaussianFiltering

# Packages
using DataFrames, CSV
using Distributions
 using Plots
using ActionModels, HierarchicalGaussianFiltering

#
# common_prob
## Probability of the world generating a common cause
## This is analogous to the brain estimating a state of whether to bind or not to bind stimuli
# Degree vector
## Min and Max degrees along the azimuth


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
    n_timesteps, 
    prob_common, 
    noise_std, 
    signal_noise_std, 
    azimuth_min, 
    azimuth_max,
    starting_position="shared")
    # Initialize the common and independent sources at random positions
    
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
        
        if is_common
            println("Timestep $t: Common Source Active")
            # Update the common source position with Gaussian noise
            common_source = move_source(common_source, noise_std, azimuth_min, azimuth_max)
            
            # Sample auditory and visual signals from the common source with noise
            auditory_signal = sample_signal(common_source)
            visual_signal = sample_signal(common_source)
            
            println("Common Source Position: ", common_source.position)
            println("Auditory Signal: ", auditory_signal)
            println("Visual Signal: ", visual_signal)
        else
            println("Timestep $t: Independent Sources Active")
            # Update independent sources
            independent_light = move_source(independent_light, noise_std, azimuth_min, azimuth_max)
            independent_sound = move_source(independent_sound, noise_std, azimuth_min, azimuth_max)
            
            # Sample auditory and visual signals from independent sources
            auditory_signal = sample_signal(independent_sound)
            visual_signal = sample_signal(independent_light)
            
            println("Independent Light Source Position: ", independent_light.position)
            println("Independent Sound Source Position: ", independent_sound.position)
            println("Auditory Signal: ", auditory_signal)
            println("Visual Signal: ", visual_signal)
        end

        # Store the auditory and visual signals along with the flag for common source
        signal_data[t] = ([auditory_signal, visual_signal], is_common)
    end

    return signal_data
end


# Initialize the simulation parameters
azimuth_min = -22.0
azimuth_max = 22.0
n_timesteps = 2000
prob_common = 0.3
noise_std = 2.5 # Noise for random walk
signal_noise_std = 1  # Noise around the signal for both common and independent sources

is_common = rand() < prob_common
print(is_common)


# Run the simulation and store results
signal_data = simulate_experiment(n_timesteps, 
                                    prob_common, 
                                    noise_std, 
                                    signal_noise_std, 
                                    azimuth_min,
                                    azimuth_max)

# Print the results
for (t, (signals, is_common)) in enumerate(signal_data)
    source_type = is_common ? "Common" : "Independent"
    println("Timestep $t: Auditory Signal = $(signals[1]), Visual Signal = $(signals[2]), Source Type = $source_type")
end

signal_data

"""
Plotting the Results
"""

using Plots

# Prepare data for plotting
auditory_common = []
auditory_independent = []
visual_common = []
visual_independent = []
timestamps = 1:n_timesteps

for (signals, is_common) in signal_data
    if is_common
        # Push actual values to the common signals and NaN to the independent signals
        push!(auditory_common, signals[1])
        push!(visual_common, signals[2])
        push!(auditory_independent, NaN)
        push!(visual_independent, NaN)
    else
        # Push actual values to the independent signals and NaN to the common signals
        push!(auditory_independent, signals[1])
        push!(visual_independent, signals[2])
        push!(auditory_common, NaN)
        push!(visual_common, NaN)
    end
end


# Define colors and markers
colors = [:blue, :red] # Blue for common, Red for independent
markers = [:square, :triangle] # Square for common, Triangle for independent

# Plot the data
p1 = plot(timestamps, auditory_common, seriestype = :scatter, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
plot!(p1, timestamps, auditory_independent, seriestype = :scatter, color = colors[1], marker = markers[2], label = "Independent Auditory")
plot!(p1, timestamps, visual_common, seriestype = :scatter, color = colors[2], marker = markers[1], label = "Common Visual")
plot!(p1, timestamps, visual_independent, seriestype = :scatter, color = colors[2], marker = markers[2], label = "Independent Visual")

plot(p1, xlabel = "Timestep", ylabel = "Signal Location", title = "Auditory and Visual Signals")

# 2. seperate plot
# Plot the data
p1 = plot(timestamps, auditory_independent, seriestype = :line, color = colors[1], marker = markers[1], label = "Independent Auditory", legend = :topright)
plot!(p1, timestamps, visual_independent, seriestype = :line, color = colors[2], marker = markers[2], label = "Independent Visual")

p2 = plot(timestamps, auditory_common, seriestype = :line, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
plot!(p2, timestamps, visual_common, seriestype = :line, color = colors[2], marker = markers[2], label = "Common Visual")

# Combine plots
plot(p1, p2, layout = (2, 1), xlabel = "Timestep", ylabel = "Signal Location", title = "Auditory and Visual Signals")

visual_common
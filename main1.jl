using Plots
using .simulator_1

# Initialize the simulation parameters
azimuth_min = -22.0
azimuth_max = 22.0
n_timesteps = 2000
prob_common = 0.3
noise_std = 2.5  # Noise for random walk
signal_noise_std = 1  # Noise around the signal for both common and independent sources

# Run the simulation and store results
signal_data = Simulator.simulate_experiment(
    n_timesteps, 
    prob_common, 
    noise_std, 
    signal_noise_std, 
    azimuth_min,
    azimuth_max
)

# Prepare data for plotting
auditory_common = []
auditory_independent = []
visual_common = []
visual_independent = []
timestamps = 1:n_timesteps

for (signals, is_common) in signal_data
    if is_common
        push!(auditory_common, signals[1])
        push!(visual_common, signals[2])
        push!(auditory_independent, NaN)
        push!(visual_independent, NaN)
    else
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
p1 = plot(timestamps, auditory_common, seriestype = :line, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
plot!(p1, timestamps, auditory_independent, seriestype = :line, color = colors[1], marker = markers[2], label = "Independent Auditory")
plot!(p1, timestamps, visual_common, seriestype = :line, color = colors[2], marker = markers[1], label = "Common Visual")
plot!(p1, timestamps, visual_independent, seriestype = :line, color = colors[2], marker = markers[2], label = "Independent Visual")

plot(p1, xlabel = "Timestep", ylabel = "Signal Location", title = "Auditory and Visual Signals")

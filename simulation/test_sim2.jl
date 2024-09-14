include("simulator_2.jl")
using Plots
using .simulator_2

# Parameters
n_timesteps = 2000
azimuth_min = -22.0
azimuth_max = 22.0
prob_common = 0.3
noise_std = 2.5
signal_noise_std = 1.0
initial_bound_std = 1.0
boundary_timesteps = [500, 1000, 1500]
boundary_std_values = [1.0, 10.0, 1.5]


# Run the simulation
signal_data, boundary_std_data = simulator_2.simulate_experiment_with_discrete_boundary(
    n_timesteps, 
    prob_common, 
    noise_std, 
    signal_noise_std, 
    azimuth_min,
    azimuth_max,
    initial_bound_std,
    boundary_timesteps,
    boundary_std_values
)

# Prepare data for plotting
auditory_common = []
auditory_independent = []
visual_common = []
visual_independent = []
timestamps = 1:n_timesteps

# for plotting boundary intervals
bounds_low = []
bounds_high = []
bounds = []
common_source_position_array = []

boundary_std_array = []

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

# Calculate the bounds and clamp them
for (bound_std, common_source_position) in boundary_std_data
    push!(common_source_position_array, common_source_position)
    
    # Calculate bounds and ensure they are within the azimuth limits
    low_bound = clamp(common_source_position - bound_std, azimuth_min, azimuth_max)
    high_bound = clamp(common_source_position + bound_std, azimuth_min, azimuth_max)
    
    push!(bounds_low, low_bound)
    push!(bounds_high, high_bound)

    push!(boundary_std_array, bound_std)
end

print(bounds_low)

# Define colors and markers
colors = [:blue, :red] # Blue for common, Red for independent
markers = [:square, :triangle] # Square for common, Triangle for independent

# Create plots for signals
p1 = plot(timestamps, auditory_independent, seriestype = :line, color = colors[1], marker = markers[1], label = "Independent Auditory", legend = :topright)
plot!(p1, timestamps, visual_independent, seriestype = :line, color = colors[2], marker = markers[2], label = "Independent Visual")

p2 = plot(timestamps, auditory_common, seriestype = :line, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
plot!(p2, timestamps, visual_common, seriestype = :line, color = colors[2], marker = markers[2], label = "Common Visual")

# Add shaded region (using ribbon to shade the boundary)
p3 = plot(timestamps, common_source_position_array, ribbon = (bounds_high, bounds_low), 
    fillalpha = 0.2, color = :blue, label = "Boundary Std Dev")

# Additional plot for boundary standard deviation over time
p4 = plot(timestamps, boundary_std_array, seriestype = :line, color = :green, label = "Boundary Std Dev", legend = :topright, title="Boundary Std Over Time")

# Combine plots
plot(p1, p2, p3, p4, layout = (4, 1), xlabel = "Timestep", ylabel = "Signal Location", title = "Auditory and Visual Signals with Dynamic Bound", size = (800, 800))
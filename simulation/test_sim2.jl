include("simulator_2.jl")
using Plots
using .simulator_2

# Parameters
n_timesteps = 3000
azimuth_min = -22.0
azimuth_max = 22.0
prob_common = 0.3
noise_std = 2.5
signal_noise_std = 1.0
initial_bound_std = 1.0
boundary_timesteps = [500, 1000, 1500, 2500]
boundary_std_values = [3.0, 1.0, 10, 1]

# Ensure boundary_timesteps and boundary_std_values are of the same length
if length(boundary_timesteps) != length(boundary_std_values)
    error("boundary_timesteps and boundary_std_values must have the same length")
end

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
p1 = plot(timestamps, auditory_independent, seriestype = :scatter, color = colors[1], marker = markers[2], label = "Independent Auditory", legend = :topright)
plot!(p1, timestamps, visual_independent, seriestype = :scatter, color = colors[2], marker = markers[2], label = "Independent Visual",
        xlabel="Timestep",
        ylabel="Signal Location",
        title = "Independent auditory and visual signals")

p2 = plot(timestamps, auditory_common, seriestype = :scatter, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
plot!(p2, timestamps, visual_common, seriestype = :scatter, color = colors[2], marker = markers[1], label = "Common Visual",
        xlabel="Timestep",
        ylabel="Signal Location",
        title = "Common auditory and visual signals")

# Add shaded region (using ribbon to shade the boundary)
#=p3 = plot(timestamps, common_source_position_array, ribbon = (bounds_high, bounds_low), 
    fillalpha = 0.2, color = :blue, label = "Boundary Std Dev")
=#

# Additional plot for boundary standard deviation over time
p3 = plot(timestamps, boundary_std_array, seriestype = :line, color = :green, label = "Boundary Std Dev", legend = :topright, 
        title="Boundary Std Over Time",
        xlabel = "Timestep",
        ylabel = "Boundary SD")



# Create a plot for parameter annotation (empty plot with text)
parameters_text = "Parameters:\n" *
    "Timesteps: $n_timesteps\n" *
    "Azimuth Range: [$azimuth_min, $azimuth_max]\n" *
    "Probability of Common cause: $prob_common\n" *
    "Random walk Noise Std Dev: $noise_std\n" *
    "Signal (sampling) Noise Std Dev: $signal_noise_std\n" *
    "Initial Bound Std Dev: $initial_bound_std\n" *
    "Boundary Timesteps: $boundary_timesteps\n" *
    "Boundary Std Values: $boundary_std_values\n"


# Create a blank plot for parameters without axes or grid
p4 = plot(legend = false, grid = false, frame = :none, xaxis = false, yaxis = false, size=(800, 200))
annotate!(p4, (0.05, 0.5, text(parameters_text, 10, :black, halign = :left)))

# Create a blank plot for the main title
main_title_plot = plot(legend = false, grid = false, frame = :none, xaxis = false, yaxis = false, size=(800, 100))
annotate!(main_title_plot, (0.5, 0.5, 
            text("Simulation with dynamic boundary", 
            15, :black, halign = :center, valign = :center)))
annotate!(main_title_plot, (0.5, 0.1, 
            text("If independent cause is picked, A and V signals are sampled from a gaussian boundary around
            the common cause source", 
            10, :black, halign = :center, valign = :center)))

# Combine all plots with the title at the top
p = plot(main_title_plot, p1, p2, p3, p4, layout = (5, 1), size = (800, 900))

display(p)

# homemade functions
include("plot_sim1.jl")
include("simulator_1.jl")

using .simulator_1
using .Plotter_1

# store-bought functions
using Plots


# Initialize the simulation parameters
azimuth_min = -22.0
azimuth_max = 22.0
n_timesteps = 2000
prob_common = 0.3
noise_std = 2.5  # Noise for random walk
signal_noise_std = 1.0  # Noise around the signal for both common and independent sources

# Run the simulation and store results
signal_data = simulator_1.simulate_experiment(
    n_timesteps, 
    prob_common, 
    noise_std, 
    signal_noise_std, 
    azimuth_min,
    azimuth_max
)

# Call the modular plotting function from Plotter
Plotter.plot_simulation1_results(signal_data, n_timesteps)
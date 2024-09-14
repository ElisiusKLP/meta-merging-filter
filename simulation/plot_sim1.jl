module Plotter_1
using Plots

function plot_simulation1_results(
    signal_data::Vector{Tuple{Vector{Float64}, Bool}}, 
    n_timesteps::Int
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

    colors = [:blue, :red] # Blue for auditory, Red for visual
    markers = [:square, :circle] # Square for common, circle for independent

    p1 = plot(timestamps, auditory_independent, seriestype = :line, color = colors[1], marker = markers[2], label = "Independent Auditory", legend = :topright)
    plot!(p1, timestamps, visual_independent, seriestype = :line, color = colors[2], marker = markers[2], label = "Independent Visual")

    p2 = plot(timestamps, auditory_common, seriestype = :line, color = colors[1], marker = markers[1], label = "Common Auditory", legend = :topright)
    plot!(p2, timestamps, visual_common, seriestype = :line, color = colors[2], marker = markers[1], label = "Common Visual")

    # Combine plots
    plot(p1, p2, layout = (2, 1), xlabel = "Timestep", ylabel = "Signal Location", title = "Auditory and Visual Signals")
end

end

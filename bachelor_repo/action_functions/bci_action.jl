""" 
Original BCI model action. Translated to action function.
"""

original_action_model = function(
    agent::Agent, 
    input, constant_cue = "A", 
    internal_variables = "clean",
    decision = "model_averaging", 
    )
    """ 
    constant_cue can be either "A" or "V" and is used to specify whether the cue is auditory or visual
    decision can be either "model_averaging" or "model_selection" and is used to specify whether the decision rule is model averaging or model selection
    """

    # generate noise
    auditory_latent_noise = randn(1)[1]
    visual_latent_noise = randn(1)[1]

    if internal_variables == "noisy"
        if length(input) == 3
            auditory_stimulus = input[1] * auditory_latent_noise
            visual_stimulus = input[2] * visual_latent_noise

            cue = input[3]
        else
            auditory_stimulus = input[1] * auditory_latent_noise
            visual_stimulus = input[2] * visual_latent_noise

            cue = constant_cue
        end
    elseif internal_variables == "clean"
        if length(input) == 3
            auditory_stimulus = input[1]
            visual_stimulus = input[2]

            cue = input[3]
        else
            auditory_stimulus = input[1]
            visual_stimulus = input[2]

            cue = constant_cue
        end
    end


    #Get parameters
    p_common = agent.parameters["p_common"]
    muP = agent.parameters["muP"]
    sigP = agent.parameters["sigP"]
    sigA = agent.parameters["sigA"]
    sigV = agent.parameters["sigV"]
    action_noise = agent.parameters["action_noise"]

    # variances of A and V and prior
    varP = sigP^2
    varA = sigA^2
    varV = sigV^2

    # variances of estimates given common or independent
    #= varVA_hat = 1 / ( 1 / varV + 1 / varA + 1 / varP )
    varV_hat = 1 / ( 1 / varV + 1 / varP )
    varA_hat = 1 / ( 1 / varA + 1 / varP ) =#
    # I drop this as it is just confusing code practice when realting it to the used equations in papers
    # It is instead part of the sAV_hat_if_common equation

    # variances used in computing probasbility of common or independent causes
    var_common = varV * varA + varV * varP + varA * varP
    varV_independent = varV + varP
    varA_independent = varA + varP

    # Calculate estimates sAV and sA and sV (forces fusion and segreated)
    # både for common og ikke common
    sAV_hat_if_common = ( (auditory_stimulus / varA) + (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters

    sA_hat_if_common = sAV_hat_if_common
    sV_hat_if_common = sAV_hat_if_common
    
    S_A_if_independent = ( (auditory_stimulus / varA) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varP) ) # everything is either observations or parameters
    S_V_if_independent = ( (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters
    
    # udregn prob of common or independent
    ## this is a weighted distance metric
    quad_common = ( visual_stimulus - auditory_stimulus )^2 * varP + ( visual_stimulus - muP )^2 * varA + ( auditory_stimulus - muP )^2 * varV
    quadV_independent = ( visual_stimulus - muP )^2
    quadA_independent = ( auditory_stimulus - muP )^2

    # likelihood of observations (xV, xA) given C (1=common or 2=independent)
    ## this is the PDF
    likelihood_common = exp(-quad_common/(2*var_common)) / (2*pi*sqrt(var_common))
    likelihoodV_independent = exp(-quadV_independent/(2*varV_independent)) / sqrt(2*pi*varV_independent)
    likelihoodA_independent = exp(-quadA_independent/(2*varA_independent)) / sqrt(2*pi*varA_independent)
    likelihood_independent = likelihoodV_independent * likelihoodA_independent

    # posterior probability of state C (cause 1 or 2) given observations (xV, xA)
    posterior_common = likelihood_common * p_common
    posterior_independent = likelihood_independent * (1 - p_common)
    posterior_C = posterior_common / ( posterior_common + posterior_independent )
    
    # DECISION RULE
    if decision == "model_averaging"
        sV_hat = posterior_C * sV_hat_if_common + (1 - posterior_C) * S_V_if_independent
        sA_hat = posterior_C * sA_hat_if_common + (1 - posterior_C) * S_A_if_independent
    elseif decision == "model_selection"
        sV_hat = (posterior_C > 0.5) * sV_hat_if_common + (posterior_C <= 0.5) * S_V_if_independent
        sA_hat = (posterior_C > 0.5) * sA_hat_if_common + (posterior_C <= 0.5) * S_A_if_independent
    end

    #update states
    push!(agent.history["C"], posterior_C)

    if cue == "A"
        action = Normal(sA_hat, action_noise)
    elseif cue == "V"
        action = Normal(sV_hat, action_noise)
    end

    
    return action
end
"""
Merging HGF function
"""

function merging_hgf(
    agent::Agent, 
    input, 
    constant_cue="A", 
    decision="model_averaging_surprise",
    reset = "no_reset",
    )

    if length(input) == 5
        FF_auditory_stimulus = input[1]
        FF_visual_stimulus = input[2]

        Seg_auditory_stimulus = input[3]
        Seg_visual_stimulus = input[4]

        cue = input[5]
    else
        FF_auditory_stimulus = input[1]
        FF_visual_stimulus = input[2]

        Seg_auditory_stimulus = input[3]
        Seg_visual_stimulus = input[4]

        cue = constant_cue
    end

    action_noise = agent.parameters["action_noise"]
    prior_common = agent.parameters["p_common"]

    hgf = agent.substruct

    # update HGF with inputs
    update_hgf!(hgf, [FF_auditory_stimulus, FF_visual_stimulus, Seg_auditory_stimulus, Seg_visual_stimulus])

    # Forced Fusion (Common Cause)
    FF_inferred_location = get_states(hgf, ("FF_sAV", "posterior_mean"))

    FF_action = Normal(FF_inferred_location, action_noise)
    ###FF_action_distribution = Normal(FF_inferred_location, action_noise)

    # Segregation (Independent Causes)
    Ind_inferred_auditory_location = get_states(hgf, ("Seg_sA", "posterior_mean"))
    Ind_inferred_visual_location = get_states(hgf, ("Seg_sV", "posterior_mean"))

    Ind_action_auditory = Normal(Ind_inferred_auditory_location, action_noise)
    Ind_action_visual = Normal(Ind_inferred_visual_location, action_noise)

    # DECISION
    ## Model averaging w. surprise
    if decision == "model_averaging_surprise"
            
        ## Model Averaging using surprise
        surpriseA_common_cause = get_surprise(hgf, ("FF_A"))
        surpriseV_common_cause = get_surprise(hgf, ("FF_V"))
        surprise_common_cause = surpriseA_common_cause + surpriseV_common_cause

        surpriseA_independent = get_surprise(hgf, ("Seg_A"))
        surpriseV_independent = get_surprise(hgf, ("Seg_V"))
        surprise_independent = surpriseA_independent + surpriseV_independent

        #print("surprises: $surprise_common_cause and $surprise_independent")
        #print("likelihoods: $(-exp(surprise_common_cause)) and $(-exp(surprise_independent))")

        posterior_common = -exp(surprise_common_cause) * prior_common
        posterior_independent = -exp(surprise_independent) * (1 - prior_common)
        posterior_C = posterior_common / ( posterior_common + posterior_independent )

        push!(agent.history["C"], posterior_C)
        
        #print("posterior_common: $posterior_common")
        #print("posterior_independent: $posterior_independent")

        #print("inputs: ", input)
        #print("posterior_C: $posterior_C")
        
        #soft_array = [surprise_common_cause, surprise_independent]
        #softmax = softmax!(soft_array)
        
        # this output the exact same as the posterior_c
        #print("softmax: $softmax") 

        # I call this posterior even though i haven't multiplied by the prior
        # I asumme the model accumulates evidence into the suprise information
        #posterior_common = softmax[1]
        #posterior_independent = softmax[2]

        #cause_probability_ratio = posterior_common / posterior_independent
        
        # MODEL AVERAGING
        sA_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_auditory_location
        sV_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_visual_location

        #print("sV_hat: $sV_hat")
        #print("sA_hat: $sA_hat")
        

    elseif decision == "model_averaging"

        #FF_likelihood = pdf
           # CAUSE estimation
        ## calculating likelihoods

    
        
        FF_likelihood = pdf(FF_action, FF_auditory_stimulus) * pdf(FF_action, FF_visual_stimulus)
        Ind_likelihood_auditory = pdf(Ind_action_auditory, Seg_auditory_stimulus)
        Ind_likelihood_visual = pdf(Ind_action_visual, Seg_visual_stimulus)
        likelihood_independent = Ind_likelihood_auditory * Ind_likelihood_visual
        
        posterior_common = FF_likelihood * prior_common
        posterior_independent = likelihood_independent * (1 - prior_common)
        posterior_C = posterior_common / ( posterior_common + posterior_independent )
        
        print("inputs: ", input)
        print("posterior_C: $posterior_C")

        # MODEL AVERAGING
        sA_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_auditory_location
        sV_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_visual_location

        print("sA_hat: $sV_hat")
        print("sV_hat: $sA_hat")
        
    end
    
    ## 

    # Generating action
    if cue == "A"
        action = Normal(sA_hat, action_noise)
    elseif cue == "V"
        action = Normal(sV_hat, action_noise)
    end

    if reset == "reset" 
        reset!(hgf)
    end
    
    return action
end


"""
Merging HGF function
"""

function merging_hgf_w_reset(
    agent::Agent, 
    input, 
    constant_cue="A", 
    internal_variables = "clean", 
    decision="model_averaging_surprise",
    reset = "reset",
    )

    # generate latent noise for the internal variables
    auditory_latent_noise = randn(1)[1]
    visual_latent_noise = randn(1)[1]

    if internal_variables == "noisy"
        if length(input) == 5
            FF_auditory_stimulus = input[1] * auditory_latent_noise
            FF_visual_stimulus = input[2] * visual_latent_noise

            Seg_auditory_stimulus = input[3] * auditory_latent_noise
            Seg_visual_stimulus = input[4] * visual_latent_noise

            cue = input[5]
        else
            FF_auditory_stimulus = input[1] * auditory_latent_noise
            FF_visual_stimulus = input[2] * visual_latent_noise

            Seg_auditory_stimulus = input[3] * auditory_latent_noise
            Seg_visual_stimulus = input[4] * visual_latent_noise

            cue = constant_cue
        end
    elseif internal_variables == "clean"
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
    end

    action_noise = agent.parameters["action_noise"]
    prior_common = agent.parameters["p_common"]

    hgf = agent.substruct

    # update HGF with inputs
    update_hgf!(hgf, [FF_auditory_stimulus, FF_visual_stimulus, Seg_auditory_stimulus, Seg_visual_stimulus])

    # Forced Fusion (Common Cause)
    FF_inferred_location = get_states(hgf, ("FF_sAV", "posterior_mean"))

    ###FF_action_distribution = Normal(FF_inferred_location, action_noise)

    # Segregation (Independent Causes)
    if cue == "A"
        Ind_inferred_location = get_states(hgf, ("Seg_sA", "posterior_mean"))
    elseif cue == "V"
        Ind_inferred_location = get_states(hgf, ("Seg_sV", "posterior_mean"))
    end

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

        posterior_common = -exp(surprise_common_cause) * prior_common
        posterior_independent = -exp(surprise_independent) * (1 - prior_common)
        posterior_C = posterior_common / ( posterior_common + posterior_independent )
        
        #=soft_array = [-surprise_common_cause, -surprise_independent]
        softmax = softmax!(soft_array)
        
        # this output the exact same as the posterior_c
        print("softmax: $softmax") =#

        # I call this posterior even though i haven't multiplied by the prior
        # I asumme the model accumulates evidence into the suprise information
        #posterior_common = softmax[1]
        #posterior_independent = softmax[2]

        #cause_probability_ratio = posterior_common / posterior_independent
        
        # MODEL AVERAGING
        sV_hat = posterior_common * FF_inferred_location + posterior_independent * Ind_inferred_location
        sA_hat = posterior_common * FF_inferred_location + posterior_independent * Ind_inferred_location
    elseif decision == "model_averaging"

        FF_likelihood = pdf
           # CAUSE estimation
    ## calculating likelihoods

    #FF_likelihood_auditory = pdf(MvNormal(FF_inferred_location, FF_inferred_location]))
 #= 
    Ind_likelihood_auditory = pdf(Ind_action_distribution, auditory_stimulus)
    Ind_likelihood_visual = pdf(Ind_action_distribution, visual_stimulus)
    likelihood_independent = Ind_likelihood_auditory * Ind_likelihood_visual
 =#
        
        # MODEL AVERAGING
        sV_hat = prior_common * FF_inferred_location + (1 - prior_common) * Ind_inferred_location
        sA_hat = prior_common * FF_inferred_location + (1 - prior_common) * Ind_inferred_location
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


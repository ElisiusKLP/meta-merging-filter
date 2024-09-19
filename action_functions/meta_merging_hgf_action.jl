"""
Merging HGF function
"""

function meta_merging_hgf_action(
    agent::Agent, 
    input, 
    constant_cue="A", 
    internal_variables = "clean",
    meta_network = "meta1",
    decision="model_averaging",
    reset = "no_reset",
    )

    if internal_variables == "clean"
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

    hgf = agent.substruct

    # update HGF with inputs
    update_hgf!(hgf, [FF_auditory_stimulus, FF_visual_stimulus, Seg_auditory_stimulus, Seg_visual_stimulus, missing, missing])
    #update_hgf!(hgf, [FF_auditory_stimulus, FF_visual_stimulus, missing, missing, missing, missing])

    # Forced Fusion (Common Cause)
    FF_inferred_location = get_states(hgf, ("FF_sAV", "posterior_mean"))

    ###FF_action_distribution = Normal(FF_inferred_location, action_noise)

    # Segregation (Independent Causes)
    if cue == "A"
        Ind_inferred_location = get_states(hgf, ("Seg_sA", "posterior_mean"))
    elseif cue == "V"
        Ind_inferred_location = get_states(hgf, ("Seg_sV", "posterior_mean"))
    end

    # META HGF
    # i think the goal is to pipe the surprise for each cause
    # into the meta HGF and make it

    # Meta Network 1: Piping of the model posterior of the common cause piped to the meta hgf
    ## We compute posteriors
    if meta_network == "meta1"
        surpriseA_common_cause = get_surprise(hgf, ("FF_uA"))
        surpriseV_common_cause = get_surprise(hgf, ("FF_uV"))
        surprise_common_cause = surpriseA_common_cause + surpriseV_common_cause

        surpriseA_independent = get_surprise(hgf, ("Seg_uA"))
        surpriseV_independent = get_surprise(hgf, ("Seg_uV"))
        surprise_independent = surpriseA_independent + surpriseV_independent

        print("surprises: common_cause: $surprise_common_cause and independent: $surprise_independent")
        print("likelihoods: common_cause: $(-exp(surprise_common_cause)) and independent: $(-exp(surprise_independent))")
        
        likelihood_common = -exp(surprise_common_cause) #inverse surprise for sum of input node surprises
        likelihood_independent = -exp(surprise_independent)

        #update_hgf!(hgf, [missing, missing, missing, missing, likelihood_common, likelihood_independent])

        posterior_common = get_states(hgf, ("Meta_sCOM", "posterior_mean"))
        posterior_independent = get_states(hgf, ("Meta_sIND", "posterior_mean"))

        print("posterior_common: $posterior_common")
        print("posterior_independent: $posterior_independent")

        posterior_C = posterior_common / ( posterior_common + posterior_independent )

        print("posterior_C: $posterior_C")
    end
    
    # 3. DECISION
    if decision == "model_averaging"
        # MODEL AVERAGING
        sA_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_location
        sV_hat = posterior_C * FF_inferred_location + (1 - posterior_C) * Ind_inferred_location
    end

    # Generating action
    if cue == "A"
        #action = sA_hat
        action = Normal(sA_hat, action_noise)
    elseif cue == "V"
        #action = sV_hat
        action = Normal(sV_hat, action_noise)
    end

    if reset == "reset" 
        reset!(hgf)
    end
    
    return action
end


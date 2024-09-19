
using HierarchicalGaussianFiltering

nodes = [
    # SEGREGATION
	ContinuousInput(
		name = "seg_uA",
		input_noise = -2,
		bias = 0
	),
	ContinuousState(
		name = "seg_sA",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "seg_uV",
		input_noise = -2,
		bias = 0
	),
	ContinuousState(
		name = "seg_sV",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
    # FORCED FUSION
	ContinuousState(
		name = "ff_sAV",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "ff_uA",
		input_noise = -2,
		bias = 0
	),
	ContinuousInput(
		name = "ff_uV",
		input_noise = -2,
		bias = 0
	),
    # META HGF
    ContinuousState(
		name = "meta_s",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	),
	ContinuousInput(
		name = "meta_u",
		input_noise = -2,
		bias = 0
	),
	ContinuousState(
		name = "meta_vop",
		volatility = -2,
		drift = 0,
		autoconnection_strength = 1,
		initial_mean = 0,
		initial_precision = 1
	)
]

edges = Dict(
	("seg_uA", "seg_sA") => ObservationCoupling(),
	("seg_uV", "seg_sV") => ObservationCoupling(),
	("ff_uA", "ff_sAV") => ObservationCoupling(),
	("ff_uV", "ff_sAV") => ObservationCoupling(),
	("meta_u", "meta_s") => ObservationCoupling(),
	("meta_s", "meta_vop") => DriftCoupling(1, LinearTransform())
)

#CHANGE THIS TO THE CORRECT ORDER
update_order = ["seg_uA", "seg_sA", "seg_uV", "seg_sV", "ff_uA", "ff_uV", "ff_sAV", "meta_u", "meta_s", "meta_vop"]

my_network = init_hgf(
    nodes = nodes,
    edges = edges,
    update_order = update_order
)

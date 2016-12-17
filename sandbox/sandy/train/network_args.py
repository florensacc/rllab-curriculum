import lasagne.nonlinearities as NL

icml_trpo_atari_args = dict(
    conv_filters=[16,16],
    conv_filter_sizes=[4,4],
    conv_strides=[2,2],
    conv_pads=[(0,0)]*2,
    hidden_sizes=[20],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.softmax,
)

nips_dqn_args=dict(
    conv_filters=[16,32],
    conv_filter_sizes=[8,4],
    conv_strides=[4,2],
    conv_pads=[(0,0)]*2,
    hidden_sizes=[256],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.softmax,
)

baseline_args = dict(
    hidden_sizes=[16,16],
    conv_filters=[],
    conv_filter_sizes=[],
    conv_strides=[],
    conv_pads=[],
    #batchsize=30000,  # 6000
)

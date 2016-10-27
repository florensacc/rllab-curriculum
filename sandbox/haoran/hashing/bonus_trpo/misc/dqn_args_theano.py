import lasagne.nonlinearities as NL

trpo_dqn_args=dict(
    conv_filters=[16,16],
    conv_filter_sizes=[4,4],
    conv_strides=[2,2],
    conv_pads=[(0,0)]*2,
    hidden_sizes=[20],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.softmax,
)

nature_dqn_args=dict(
    conv_filters=[32,64,64],
    conv_filter_sizes=[8,4,3],
    conv_strides=[4,2,1],
    conv_pads=[(0,0)]*3,
    hidden_sizes=[512],
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

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

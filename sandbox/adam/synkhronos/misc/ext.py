
from rllab.misc.console import Message


def compile_synk_function(inputs=None, outputs=None, updates=None, givens=None,
        log_name=None, collect_modes=None, reduce_ops=None, **kwargs):
    import synkhronos
    if log_name:
        msg = Message("Compiling Synkhronos function %s" % log_name)
        msg.__enter__()
    ret = synkhronos.function(
        inputs=inputs,
        outputs=outputs,
        updates=updates,
        givens=givens,
        collect_modes=collect_modes,
        reduce_ops=reduce_ops,
        on_unused_input='ignore',
        allow_input_downcast=True,
        **kwargs
    )
    if log_name:
        msg.__exit__(None, None, None)
    return ret

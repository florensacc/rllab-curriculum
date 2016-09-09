

from theano.tensor.opt import register_canonicalize
import theano


class CustomGrad(theano.compile.ViewOp):
    def make_node(self, x, known):
        return theano.gof.Apply(self, [x, known], [x.type()])

    def perform(self, node, inp, out):
        x, _ = inp
        z, = out
        z[0] = x

    def c_code(self, node, nodename, inp, out, sub):
        # import ipdb; ipdb.set_trace()
        iname, _ = inp
        oname, = out
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        return super(CustomGrad, self).c_code(node, nodename, inp, out, sub)

    def grad(self, args, g_outs):
        return [g_outs[0], g_outs[0]]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


custom_grad = CustomGrad()
register_canonicalize(theano.gof.PatternSub((custom_grad, 'x', 'y'), 'x'), name='remove_custom_grad')

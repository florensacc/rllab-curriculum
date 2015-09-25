from __future__ import absolute_import
import click
from pydoc import locate

class ClassParamType(click.ParamType):

    name = 'path'

    def convert(self, value, param, ctx):
        mod = locate(value)
        if mod:
            return mod
        self.fail('Cannot load class from %s' % value, param, ctx)

CLASS = ClassParamType()

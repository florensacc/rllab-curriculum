import os
import re
import subprocess
import tempfile
from collections import OrderedDict

import pycparser
from pycparser.c_ast import ArrayDecl, TypeDecl, PtrDecl


def tryint(x):
    try:
        return int(x)
    except:
        return x


HEADER_DIR = os.path.expanduser('~/.mujoco/mjpro140/include')
HEADER_FILES = [
    'mjmodel.h',
    'mjdata.h',
    'mjvisualize.h',
    'mjrender.h',
]
OUTPUT = 'cymj/wrappers.pxi'
SKIP_STRUCTS = ['mjVisual']

# ===== Read all header files =====
file_contents = []
for filename in HEADER_FILES:
    with open(os.path.join(HEADER_DIR, filename), 'r') as f:
        file_contents.append(f.read())
full_src_lines = [line.strip() for line in '\n'.join(file_contents).splitlines()]

#  ===== Parse array shape hints =====
array_shapes = {}
curr_struct_name = None
for line in full_src_lines:
    # Current struct name
    m = re.match(r'struct (\w+)', line)
    if m:
        curr_struct_name = m.group(1)
        continue
    # Pointer with a shape comment
    m = re.match(r'\s*\w+\s*\*\s+(\w+);\s*//.*\((.+) x (.+)\)$', line)
    if m:
        name = curr_struct_name[1:] + '.' + m.group(1)
        assert name not in array_shapes
        array_shapes[name] = (tryint(m.group(2)), tryint(m.group(3)))

# ===== Preprocess header files =====
with tempfile.NamedTemporaryFile(suffix='.h') as f:
    f.write('\n'.join(full_src_lines).encode())
    f.flush()
    # -E: run preprocessor only
    # -P: don't generate debug lines starting with #
    # -I: include directory
    processed_src = subprocess.check_output(['cc', '-E', '-P', '-I', HEADER_DIR, f.name]).decode()

# ===== Parse and extract structs =====
ast = pycparser.c_parser.CParser().parse(processed_src)
structs = OrderedDict()

for node in ast.children():
    assert (node[1].name is None) == isinstance(node[1].type, pycparser.c_ast.Struct)
    if isinstance(node[1].type, pycparser.c_ast.Struct):
        (_, struct), = node[1].children()

        assert struct.name.startswith('_mj')
        struct_name = struct.name[1:]  # take out leading underscore
        assert struct_name not in structs
        if struct_name in SKIP_STRUCTS:
            print('Skipping {}'.format(struct_name))
            continue

        structs[struct_name] = {'scalars': [], 'arrays': [], 'ptrs': [], 'depends_on_model': False}

        for child in struct.children():
            child_name, child_type, ((_, decl),) = child[1].name, child[1].type, child[1].children()
            # print('   ', child_name, child_type)

            if isinstance(child_type, ArrayDecl):
                array_type = ' '.join(decl.type.type.names)
                if isinstance(decl.dim, pycparser.c_ast.ID):
                    array_size = decl.dim.name
                else:
                    array_size = int(decl.dim.value)
                # print('[a]    {} {}[{}]'.format(child_name, array_type, array_size))
                structs[struct_name]['arrays'].append((child_name, array_type, array_size))

            elif isinstance(child_type, TypeDecl):
                if isinstance(decl.type, pycparser.c_ast.Struct):
                    print('Warning: ignorning nested struct {}.{}'.format(struct_name, child_name))
                    # TODO: implement this (used in mjVisual)
                    continue
                field_type = ' '.join(decl.type.names)
                # print('       {} {}'.format(child_name, field_type))
                structs[struct_name]['scalars'].append((child_name, field_type))

            elif isinstance(child_type, PtrDecl):
                ptr_type = ' '.join(decl.type.type.names)
                # print('[p]    {} *{}'.format(child_name, ptr_type))
                n = struct_name + '.' + child_name
                if n not in array_shapes:
                    print('Warning: skipping {} due to unknown shape'.format(n))
                else:
                    structs[struct_name]['ptrs'].append((child_name, ptr_type, array_shapes[n]))
                    # Structs needing array shapes must get them through mjModel
                    # but mjModel itself doesn't need to be passed an extra mjModel.
                    # TODO: depends_on_model should be set to True if any member of this struct depends on mjModel
                    # but currently that never happens.
                    if struct_name != 'mjModel':
                        structs[struct_name]['depends_on_model'] = True

            else:
                raise NotImplementedError

#  ===== Generate code =====
code = []
needed_1d_wrappers = set()
needed_2d_wrappers = set()

structname2wrappername = {}
structname2wrapfuncname = {}
for name in structs:
    assert name.startswith('mj')
    structname2wrappername[name] = 'PyMj' + name[2:]
    structname2wrapfuncname[name] = 'WrapMj' + name[2:]

# ===== Generate wrapper extension classes =====
for name, fields in structs.items():
    member_decls, member_initializers, member_getters = [], [], []

    model_var_name = 'p' if name == 'mjModel' else 'model'

    for scalar_name, scalar_type in fields['scalars']:
        if scalar_type in ['float', 'int', 'mjtNum', 'mjtByte', 'unsigned int']:
            member_getters.append('    @property\n    def {name}(self): return self.ptr.{name}'.format(name=scalar_name))
            member_getters.append('    @{name}.setter\n    def {name}(self, {type} x): self.ptr.{name} = x'.format(
                name=scalar_name, type=scalar_type))
        elif scalar_type in structs:
            # This is a struct member
            member_decls.append('    cdef {} _{}'.format(structname2wrappername[scalar_type], scalar_name))
            member_initializers.append('        self._{scalar_name} = {wrap_func_name}(&p.{scalar_name}{model_arg})'.format(
                scalar_name=scalar_name,
                wrap_func_name=structname2wrapfuncname[scalar_type],
                model_arg=(', ' + model_var_name) if structs[scalar_type]['depends_on_model'] else ''
            ))
            member_getters.append('    @property\n    def {name}(self): return self._{name}'.format(name=scalar_name))
        else:
            print('Warning: skipping {} {}.{}'.format(scalar_type, name, scalar_name))

    # Pointer types
    for ptr_name, ptr_type, (shape0, shape1) in fields['ptrs']:
        if ptr_type in structs:
            assert shape0.startswith('n') and shape1 == 1
            member_decls.append('    cdef tuple _{}'.format(ptr_name))
            member_initializers.append(
                '        self._{ptr_name} = tuple([{wrap_func_name}(&p.{ptr_name}[i]{model_arg}) for i in range({size0})])'.format(
                    ptr_name=ptr_name,
                    wrap_func_name=structname2wrapfuncname[ptr_type],
                    size0='{}.{}'.format(model_var_name, shape0),
                    model_arg=(', ' + model_var_name) if structs[ptr_type]['depends_on_model'] else ''
                ))
        else:
            assert name == 'mjModel' or fields['depends_on_model']
            member_decls.append('    cdef np.ndarray _{}'.format(ptr_name))
            if shape0 == 1 or shape1 == 1:
                # Collapse to 1d for the user's convenience
                size0 = shape1 if shape0 == 1 else shape0
                member_initializers.append(
                    '        self._{ptr_name} = _wrap_{ptr_type}_1d(p.{ptr_name}, {size0})'.format(
                        ptr_name=ptr_name,
                        ptr_type=ptr_type.replace(' ', '_'),
                        size0='{}.{}'.format(model_var_name, size0) if (
                            isinstance(size0, str) and size0.startswith('n')) else size0,
                    ))
            else:
                member_initializers.append(
                    '        self._{ptr_name} = _wrap_{ptr_type}_2d(p.{ptr_name}, {size0}, {size1})'.format(
                        ptr_name=ptr_name,
                        ptr_type=ptr_type.replace(' ', '_'),
                        size0='{}.{}'.format(model_var_name, shape0) if (
                            isinstance(shape0, str) and shape0.startswith('n')) else shape0,
                        size1='{}.{}'.format(model_var_name, shape1) if (
                            isinstance(shape1, str) and shape1.startswith('n')) else shape1,
                    ))
            needed_2d_wrappers.add(ptr_type)
        member_getters.append('    @property\n    def {name}(self): return self._{name}'.format(name=ptr_name))

    # Array types: handle the same way as pointers
    for array_name, array_type, array_size in fields['arrays']:
        if array_type in structs:
            print('FIXME: skipping array', array_name, array_type)
            continue
        member_decls.append('    cdef np.ndarray _{}'.format(array_name))
        member_initializers.append(
            '        self._{array_name} = _wrap_{array_type}_1d(&p.{array_name}[0], {size})'.format(
                array_name=array_name,
                array_type=array_type.replace(' ', '_'),
                size=array_size,
            ))
        member_getters.append('    @property\n    def {name}(self): return self._{name}'.format(name=array_name))
        needed_1d_wrappers.add(array_type)

    member_getters = '\n'.join(member_getters)
    member_decls = '\n' + '\n'.join(member_decls) if member_decls else ''
    member_initializers = '\n' + '\n'.join(member_initializers) if member_initializers else ''
    model_decl = '\n    cdef mjModel* _model' if fields['depends_on_model'] else ''
    model_param = ', mjModel* model' if fields['depends_on_model'] else ''
    model_setter = '\n        self._model = model' if fields['depends_on_model'] else ''
    model_arg = ', model' if fields['depends_on_model'] else ''

    code.append('''
cdef class {wrapper_name}(object):
    cdef {struct_name}* ptr{model_decl}{member_decls}
    cdef void _set(self, {struct_name}* p{model_param}):
        self.ptr = p{model_setter}{member_initializers}
{member_getters}

cdef {wrapper_name} {wrap_func_name}({struct_name}* p{model_param}):
    cdef {wrapper_name} o = {wrapper_name}()
    o._set(p{model_arg})
    return o

    '''.format(
        wrapper_name=structname2wrappername[name],
        struct_name=name,
        wrap_func_name=structname2wrapfuncname[name],
        model_decl=model_decl,
        model_param=model_param,
        model_setter=model_setter,
        model_arg=model_arg,
        member_decls=member_decls,
        member_initializers=member_initializers,
        member_getters=member_getters,
    ).strip())

# ===== Generate array-to-NumPy wrappers =====
# TODO: instead of returning None for empty arrays, instead return NumPy arrays with the appropriate shape and type
# The only reason we're not doing this already is that cython's views don't work with 0-length axes,
# even though NumPy does.
# TODO: set NumPy array type explicitly (e.g. char will be viewed incorrectly as np.int64)
for type_name in sorted(needed_1d_wrappers):
    code.append('''
cdef inline np.ndarray _wrap_{type_name_nospaces}_1d({type_name}* a, int shape0):
    if shape0 == 0: return None
    cdef {type_name}[:] b = <{type_name}[:shape0]> a
    return np.asarray(b)
'''.format(type_name_nospaces=type_name.replace(' ', '_'), type_name=type_name).strip())

for type_name in sorted(needed_2d_wrappers):
    code.append('''
cdef inline np.ndarray _wrap_{type_name_nospaces}_2d({type_name}* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef {type_name}[:,:] b = <{type_name}[:shape0,:shape1]> a
    return np.asarray(b)
'''.format(type_name_nospaces=type_name.replace(' ', '_'), type_name=type_name).strip())

header = '''# cython: language_level=3
# Automatically generated. Do not modify!

include "mujoco.pxd"
cimport numpy as np
import numpy as np

'''

code = header + '\n\n'.join(code) + '\n'
print(len(code.splitlines()))
with open(OUTPUT, 'w') as f:
    f.write(code)

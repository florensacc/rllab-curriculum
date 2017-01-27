import distutils
import fcntl
import imp
import logging
import os
import platform
import shutil
import subprocess

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from pyximport import pyxbuild
except:
    raise Exception(
        'Failed to import pyximport.pyxbuild.\n\nHINT: Install Cython with `pip install cython`.')


class MujocoDependencyError(Exception):
    pass


def discover_mujoco():
    key_path = os.environ.get('MUJOCO_PY_MJKEY_PATH')
    if key_path and not os.path.exists(key_path):
        raise MujocoDependencyError(
            'MUJOCO_PY_MJKEY_PATH path does not exist: {}'.format(key_path))

    mjpro_path = os.environ.get('MUJOCO_PY_MJPRO140_PATH')
    if mjpro_path and not os.path.exists(mjpro_path):
        raise MujocoDependencyError(
            'MUJOCO_PY_MJPRO140_PATH path does not exist: {}'.format(mjpro_path))

    default_key_path = os.path.expanduser('~/.mujoco/mjkey.txt')
    default_mjpro_path = os.path.expanduser('~/.mujoco/mjpro140')

    if not key_path and os.path.exists(default_key_path):
        key_path = default_key_path
    if not mjpro_path and os.path.exists(default_mjpro_path):
        mjpro_path = default_mjpro_path

    if not key_path and not mjpro_path:
        raise MujocoDependencyError(
            'To use MuJoCo, you need to either populate ~/.mujoco/mjkey.txt and ~/.mujco/mjpro140, '
            'or set the MUJOCO_PY_MJKEY_PATH and MUJOCO_PY_MJPRO_PATH environment variables appropriately. '
            'Follow the instructions on https://github.com/openai/mujoco-py for where to obtain these.')
    elif not key_path:
        raise MujocoDependencyError(
            'Found your MuJoCo binaries but not license key. Please put your key into ~/.mujoco/mjkey.txt '
            'or set MUJOCO_PY_MJKEY_PATH. Follow the instructions on https://github.com/openai/mujoco-py for setup.')
    elif not mjpro_path:
        raise MujocoDependencyError(
            'Found your MuJoCo license key but not binaries for 1.4.0. '
            'GPR uses the newer mujoco 1.4.0, which you can download at https://www.roboti.us/index.html -- '
            'please put your binaries into ~/.mujoco/mjpro140, next to mjpro131. '
            'Additional instructions can be found at https://github.com/openai/mujoco-py.')

    mjpro = os.path.basename(mjpro_path)
    if mjpro != 'mjpro140':
        raise MujocoDependencyError(
            "We expected your MUJOCO_PY_MJPRO_PATH final directory to be 'mjpro140', but you provided: {} ({}). "
            "GPR uses the newer mujoco 1.4.0, which you can download at https://www.roboti.us/index.html -- "
            "please put your binaries into ~/.mujoco/mjpro140, next to mjpro131.".format(
                mjpro, mjpro_path))

    return (mjpro_path, key_path)


def load_cython_ext(mjpro_path):
    """
    Load the MjParallel Cython extension. This is safe to be called from
    multiple processes running on the same machine.
    """
    lockpath = os.path.dirname(__file__) + '/cythonlock.pyc'
    with open(lockpath, 'w') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        return unsafe_load_cython_ext_racy(mjpro_path)


def unsafe_load_cython_ext_racy(mjpro_path, skip=False):
    """
    Build and load the Cython extension. Cython only gives us back the raw
    path, regardless of whether it found a cached version or actually
    compiled. Since we do non-idempotent postprocessing of the DLL, be extra
    careful to only do that once and then atomically move to the final
    location.
    """
    debug = False
    on_mac = platform.system() == 'Darwin'

    mj_include_path = '%s/include' % mjpro_path
    mj_bin_path = '%s/bin' % mjpro_path

    def build_cext(cext_pyx_path):
        c_compiler = 'gcc'
        if distutils.spawn.find_executable(c_compiler) is None:
            raise MujocoDependencyError(
                'Could not find "%s" executable.\n\nHINT: On OS X, install GCC 6 with `brew install gcc`.')
        os.environ['CC'] = c_compiler
        os.environ['CFLAGS'] = '-O3 -I%s -I%s -fopenmp' % (np.get_include(), mj_include_path)
        if on_mac:
            gl_libs = '-lglfw.3'
        else:
            gl_libs = '%s %s -lGL' % (
                os.path.join(mj_bin_path, 'libglfw.so.3'), os.path.join(mj_bin_path, 'libglew.so'))
        os.environ['LDFLAGS'] = '-L%s -lmujoco140 %s -fopenmp' % (mj_bin_path, gl_libs)
        raw_cext_dll_path = pyxbuild.pyx_to_dll(cext_pyx_path, force_rebuild=1 if debug else 0)
        del os.environ['CFLAGS']
        del os.environ['LDFLAGS']
        del os.environ['CC']
        return raw_cext_dll_path

    def postprocess_cext(raw_cext_dll_path):
        root, ext = os.path.splitext(raw_cext_dll_path)
        final_cext_dll_path = root + '_final' + ext

        # If someone else already built the final DLL, don't bother recreating it here,
        # even though this should still be idempotent.
        if os.path.exists(final_cext_dll_path) \
                and not debug \
                and os.path.getmtime(final_cext_dll_path) >= os.path.getmtime(raw_cext_dll_path):
            return final_cext_dll_path

        tmp_final_cext_dll_path = final_cext_dll_path + '~'
        shutil.copyfile(raw_cext_dll_path, tmp_final_cext_dll_path)

        if on_mac:
            # Fix the rpath of the generated library -- i lost the Stackoverflow
            # reference here
            from_mujoco_path = '@executable_path/libmujoco140.dylib'
            to_mujoco_path = '%s/libmujoco140.dylib' % mj_bin_path
            subprocess.check_call(
                ['install_name_tool', '-change', from_mujoco_path, to_mujoco_path, tmp_final_cext_dll_path])

            from_glfw_path = 'libglfw.3.dylib'
            to_glfw_path = os.path.join(mj_bin_path, 'libglfw.3.dylib')
            subprocess.check_call(
                ['install_name_tool', '-change', from_glfw_path, to_glfw_path, tmp_final_cext_dll_path])
        else:
            if distutils.spawn.find_executable('patchelf') is None:
                raise MujocoDependencyError(
                    '`patchelf` command not found. We need this to build the Cython extension on Linux -- '
                    'see test.dockerfile for details!')
            # For some reason, under openai-runtime, this is taken, but not under
            # test.dockerfile -- should just always link against the nogl,
            # really.
            if 'libmujoco140.so' in subprocess.check_output(['ldd', tmp_final_cext_dll_path]).decode('utf-8'):
                subprocess.check_call(
                    ['/usr/local/bin/patchelf', '--remove-needed', 'libmujoco140.so', tmp_final_cext_dll_path])
            assert 'libmujoco140nogl.so' not in subprocess.check_output(['ldd', tmp_final_cext_dll_path]).decode(), \
                'Expected libmujoco140nogl not yet to be linked in -- are you overwriting an already processed library?'
            subprocess.check_call(
                ['/usr/local/bin/patchelf', '--add-needed', os.path.join(mj_bin_path, 'libmujoco140.so'), tmp_final_cext_dll_path])

            subprocess.check_call(
                ['/usr/local/bin/patchelf', '--remove-needed', 'libglfw.so.3', tmp_final_cext_dll_path])
            subprocess.check_call(
                ['/usr/local/bin/patchelf', '--add-needed', os.path.join(mj_bin_path, 'libglfw.so.3'), tmp_final_cext_dll_path])

            subprocess.check_call(
                ['/usr/local/bin/patchelf', '--remove-needed', 'libglew.so', tmp_final_cext_dll_path])
            subprocess.check_call(
                ['/usr/local/bin/patchelf', '--add-needed', os.path.join(mj_bin_path, 'libglew.so'),
                 tmp_final_cext_dll_path])

        # Do an atomic rename.
        os.rename(tmp_final_cext_dll_path, final_cext_dll_path)
        return final_cext_dll_path

    cext_pyx_path = os.path.abspath(os.path.dirname(__file__) + '/cymj.pyx')
    raw_cext_dll_path = build_cext(cext_pyx_path)
    final_cext_dll_path = postprocess_cext(raw_cext_dll_path)

    mod = imp.load_dynamic('cymj', final_cext_dll_path)
    # mod.selftest()
    return mod


mjpro_path, key_path = discover_mujoco()
cymj = load_cython_ext(mjpro_path)
cymj.activate(key_path)

# Public API:
ext_path = cymj.__file__
MjSim = cymj.MjSim
MjViewerContext = cymj.MjViewerContext
Constants = cymj.Constants
MjParallelLite = cymj.MjParallelLite

__all__ = ['ext_path', 'MjSim', 'MjViewerContext', 'Constants']

from mdp import AtariMDP
from ale_python_interface import ale_lib

mdp = AtariMDP(rom_path="vendor/atari_roms/seaquest.bin", obs_type='ram')
state, obs = mdp.sample_initial_state()

mdp._states = []

mdp._reset_ales([state])

print state.shape
print obs.shape
#print serstruct#.size

from ale_python_interface.ale_python_interface import ALEInterface
import os
import json

ale = ALEInterface()

games = ['montezuma_revenge']
base_rom_path = "sandbox/haoran/deep_q_rl/roms"

for game in games:
    game_info = dict()
    game_info_file = os.path.join(base_rom_path, game + '_info.json')
    if os.path.exists(game_info_file):
        answer = raw_input('Overwrite %s?'%(game_info_file))
        if answer in ['n','N']:
            continue
    full_rom_path = os.path.join(base_rom_path,game+'.bin')
    ale.loadROM(full_rom_path)
    game_info["min_action_set_length"] = len(ale.getMinimalActionSet())

    with open(game_info_file,'w') as f:
        json.dump(game_info, f)
        print game_info

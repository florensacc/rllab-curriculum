from sandbox.haoran.ale_python_interface.ale_python_interface \
    import ALEInterface
import os
import json

ale = ALEInterface()

games = ['venture']
base_rom_path = "sandbox/haoran/ale_python_interface/roms"

for game in games:
    game_info = dict()
    game_info_file = os.path.join(base_rom_path, game + '_info.json')
    if os.path.exists(game_info_file):
        answer = input('Overwrite %s?'%(game_info_file))
        if answer in ['n','N']:
            continue
    full_rom_path = os.path.join(base_rom_path,game+'.bin')
    ale.loadROM(str.encode(full_rom_path))
    game_info["min_action_set_length"] = len(ale.getMinimalActionSet())
    game_info["ram_size"] = ale.getRAMSize()

    with open(game_info_file,'w') as f:
        json.dump(game_info, f)
        print(game_info)

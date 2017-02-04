# from gym.spaces import prng
#
# from sandbox.rocky.new_analogy import fetch_utils
# import numpy as np
#
from sandbox.rocky.s3 import resource_manager
#
# env = fetch_utils.fetch_env()
# gpr_env = fetch_utils.get_gpr_env(env)
# policy = fetch_utils.fetch_prescribed_policy(env)
#
# paths = fetch_utils.new_policy_paths(seeds=np.arange(1000), policy=policy, env=env)
#
# import ipdb; ipdb.set_trace()
#
# site_xpos = np.concatenate([p["env_infos"]["site_xpos"] for p in paths], axis=0)
# # for idx in range(1000):
# #     env.seed = idx
# #     env.reset()
# #     site_xpos.append(np.copy(gpr_env.world.model.site_xpos))
#
# # site_xpos = np.asarray(site_xpos)
#
#
# site_names = gpr_env.world.model.site_names
# geom0_idx = site_names.index('geom0')
# geom1_idx = site_names.index('geom1')
#
# geom0_xpos = site_xpos[:, geom0_idx*3:geom0_idx*3+3]
# geom1_xpos = site_xpos[:, geom1_idx*3:geom1_idx*3+3]
#
# print(geom0_xpos.min(axis=0))
# print(geom0_xpos.max(axis=0))
# print(geom1_xpos.min(axis=0))
# print(geom1_xpos.max(axis=0))


resource_manager.register_file("fetch_relative_dagger_pretrained_3_boxes_v1.pkl", "/tmp/params.pkl")
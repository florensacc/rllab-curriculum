from rllab.sampler import parallel_sampler
parallel_sampler.config_parallel_sampler(n_parallel=4, base_seed=0)
import numpy as np
from rllab.mdp.openai_atari_mdp import AtariMDP
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.special import discount_cumsum
import joblib

discount = 0.99


mdp = AtariMDP(
    rom_name="pong",
    obs_type="ram",
    frame_skip=4
)

policy = CategoricalMLPPolicy(
    mdp,
    hidden_sizes=(32, 32),
)

parallel_sampler.populate_task(mdp, policy)

parallel_sampler.request_samples(
    policy.get_param_values(),
    max_samples=80000,
    max_path_length=4500,
    whole_paths=True
)

paths = parallel_sampler.collect_paths()

for p in paths:
    p["returns"] = discount_cumsum(p["rewards"], discount)

# observations = np.concatenate([p["observations"] for p in paths])
# actions = np.concatenate([p["actions"] for p in paths])
# returns = np.concatenate([p["returns"] for p in paths])

print "Saving data..."
joblib.dump(dict(mdp=mdp, policy=policy, paths=paths), "persist_data/atari_vf_fit.pkl", compress=3)
print "Saved!"

# np.savez_compressed("persist_data/atari_vf_fit.npz", observations=observations, actions=actions, returns=returns)

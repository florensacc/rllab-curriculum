

from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline
import argparse
import joblib
import uuid
import numpy as np

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    data = joblib.load(args.file)
    baseline = data['baseline']
    env = data['env']
    assert isinstance(baseline, ExtremeLinearBaseline)
    coeffs = baseline._coeffs
    obs_dim = env.observation_space.flat_dim
    act_dim = env.action_space.flat_dim

    try:
        env_noise_dim = baseline.env_noise_dim
    except AttributeError:
        env_noise_dim = obs_dim
    assert len(coeffs) == 2*obs_dim + 3 + 2*(baseline.lookahead - 1)*(act_dim + env_noise_dim) + 1

    split_points = np.cumsum([obs_dim, obs_dim, 1, 1, 1,
                              act_dim * (baseline.lookahead - 1),
                              act_dim * (baseline.lookahead - 1),
                              env_noise_dim * (baseline.lookahead - 1),
                              env_noise_dim * (baseline.lookahead - 1),
                              1])
    names = ['ob', 'ob2', 't', 't2', 't3', 'actnoise', 'actnoise2', 'envnoise', 'envnoise2', 'bias']
    for name, coeff in zip(names, np.split(coeffs, split_points)):
        print(name + '\n')
        print(coeff)
        print()

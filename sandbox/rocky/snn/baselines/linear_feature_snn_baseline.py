from __future__ import print_function
from __future__ import absolute_import

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
import numpy as np
import re

LATENT_RE = re.compile(r"^latent_\d+$")


class LinearFeatureSNNBaseline(LinearFeatureBaseline):
    def _features(self, path):
        state_features = LinearFeatureBaseline._features(self, path)
        # extract all latent variables
        agent_infos = path["agent_infos"]
        latents = []
        for k, v in agent_infos.iteritems():
            if LATENT_RE.match(k):
                latents.append((k, v))
        latents = [x[1] for x in sorted(latents)]
        return np.concatenate([state_features] + latents, axis=1)

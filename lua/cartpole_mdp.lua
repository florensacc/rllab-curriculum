local CartpoleMDP = torch.class('CartpoleMDP')

function CartpoleMDP:__init()
  self.py = require 'fb.python'
  self.py.exec([=[
from rllab.mdp.box2d.cartpole_mdp import CartpoleMDP
from rllab.mdp.normalized_mdp import normalize
mdp = normalize(CartpoleMDP())
  ]=])
end

function CartpoleMDP:reset()
  local ret = self.py.eval('mdp.reset()')
  return ret[1], ret[2]
end

function CartpoleMDP:step(state, action)
  local ret = self.py.eval('mdp.step(state, action)', {state = state, action = action})
  return ret[1], ret[2], ret[3], ret[4]
end

function CartpoleMDP:actionDim()
  return self.py.eval('mdp.action_dim')
end

function CartpoleMDP:observationShape()
  return self.py.eval('mdp.observation_shape')
end

function CartpoleMDP:actionBounds()
  local ret = self.py.eval('mdp.action_bounds')
  return ret[1], ret[2]
end

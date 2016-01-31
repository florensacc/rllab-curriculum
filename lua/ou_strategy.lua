local OUStrategy = torch.class('OUStrategy')

function OUStrategy:__init(mdp, theta, sigma)
  self.theta = theta
  self.sigma = sigma
  self.actionDim = mdp:actionDim()
  self.mu = torch.zeros(self.actionDim)
  self.state = nil
  self.actionLb, self.actionUb = mdp:actionBounds()
  self:episodeReset()
end

function OUStrategy:episodeReset()
  self.state = self.mu:clone()
end

function OUStrategy:_evolveState()
  local x = self.state  
  local dx = (self.mu - x) * self.theta + torch.randn(self.actionDim) * self.sigma
  self.state = x + dx
  return self.state
end

function _clip(x, lb, ub)
  local y = x:clone()
  y[torch.lt(y, lb)] = lb[torch.lt(y, lb)]
  y[torch.gt(y, ub)] = ub[torch.gt(y, ub)]
  return y
end

function OUStrategy:getAction(observation, policy)
  action = policy:getAction(observation)
  return _clip(action + self:_evolveState(), self.actionLb, self.actionUb)
end

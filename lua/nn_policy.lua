local NNPolicy = torch.class('NNPolicy')

function NNPolicy:__init(mdp, h1Size, h2Size)
  local obsDim = mdp:observationShape()[1]
  local actionDim = mdp:actionDim()

  local model = nn.Sequential()
  model:add(nn.BatchNormalization(obsDim))
  model:add(nn.Linear(obsDim, h1Size))
  model:add(nn.BatchNormalization(h1Size))
  model:add(nn.ReLU())
  model:add(nn.Linear(h1Size, h2Size))
  model:add(nn.BatchNormalization(h2Size))
  model:add(nn.ReLU())
  model:add(nn.Linear(h2Size, actionDim))
  model:add(nn.Tanh())

  self.model = model
  self.obsDim = obsDim
  self.actionDim = actionDim
end

function NNPolicy:getAction(obs)
  self.model:evaluate()
  return self.model:forward(obs:view(1, -1))[1]
end

function NNPolicy:getActions(obs)
  self.model:evaluate()
  return self.model:forward(obs)
end

function NNPolicy:forward(obs)
  self.model:training()
  return self.model:forward(obs)
end

function NNPolicy:backward(obs, err)
  self.model:training()
  return self.model:backward(obs, err)
end


function NNPolicy:getParameters()
  return self.model:getParameters()
end

function NNPolicy:zeroGradParameters()
  self.model:zeroGradParameters()
end

local NNQFunction = torch.class('NNQFunction')

function NNQFunction:__init(mdp, h1Size, h2Size)
  local obsDim = mdp:observationShape()[1]
  local actionDim = mdp:actionDim()

  --local h1Size = 100
  --local h2Size = 100

  local model1 = nn.Sequential()
  model1:add(nn.BatchNormalization(obsDim))
  model1:add(nn.Linear(obsDim, h1Size))
  model1:add(nn.BatchNormalization(h1Size))
  model1:add(nn.ReLU())

  local concat = nn.ParallelTable()
  concat:add(model1)
  concat:add(nn.Identity())

  local model = nn.Sequential()
  model:add(concat)
  model:add(nn.JoinTable(2))
  model:add(nn.Linear(h1Size + actionDim, h2Size))
  model:add(nn.ReLU())
  model:add(nn.Linear(h2Size, 1))

  --local obs = nn.Identity()()
  --local action = nn.Identity()()
  --local obs_bn = nn.BatchNormalization(obsDim)(obs)
  --local l1 = nn.Linear(obsDim, 100)(obs_bn)
  --local l1_bn = nn.BatchNormalization(100)(l1)
  --local h1 = nn.ReLU()(l1_bn)
  --local h1_action = nn.Concat(){h1, action}
  --local l2 = nn.Linear(100 + actionDim, 100)(h1_action)
  --local h2 = nn.ReLU()(l2)
  --local output = nn.Linear(1, 100)(h2)

  --local model = nn.gModule({obs, action}, {output})
  self.model = model
end

function NNQFunction:getQVal(obs, actions)
  self.model:evaluate()
  return self.model:forward({obs, actions})[{{},1}]
end

function NNQFunction:forward(obs, actions)
  self.model:training()
  return self.model:forward({obs, actions})[{{},1}]
end

function NNQFunction:zeroGradParameters()
  self.model:zeroGradParameters()
end

function NNQFunction:getParameters()
  return self.model:getParameters()
end


function NNQFunction:backward(obs, actions, err)
  self.model:training()
  return self.model:backward({obs, actions}, err:view(-1, 1))
end

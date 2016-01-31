local ReplayPool = torch.class('ReplayPool')

function ReplayPool:__init(observationDim, actionDim, maxSteps)
  self.observationDim = observationDim
  self.actionDim = actionDim
  self.maxSteps = maxSteps
  self.observations = torch.Tensor(maxSteps, observationDim)
  self.actions = torch.Tensor(maxSteps, actionDim)
  self.nextObservations = torch.Tensor(maxSteps, observationDim)
  self.rewards = torch.Tensor(maxSteps)
  self.terminals = torch.Tensor(maxSteps)
  self.bottom = 1
  self.top = 1
  self.size = 0
end

function ReplayPool:addSample(observation, action, reward, terminal)
  self.observations[self.top] = observation:view(observation:nElement())
  self.actions[self.top] = action:view(action:nElement())
  self.rewards[self.top] = reward
  -- TODO how to do proper conversion???
  self.terminals[self.top] = terminal and 1 or 0
  if self.size == maxSteps then
    self.bottom = self.bottom % self.maxSteps + 1
  else
    self.size = self.size + 1
  end
  self.top = self.top % self.maxSteps + 1
end

function ReplayPool:randomBatch(batchSize)
  local count = 0
  local indices = torch.LongTensor(batchSize)
  local transitionIndices = torch.LongTensor(batchSize)
  while count < batchSize do
    local index = torch.random(self.bottom, self.bottom + self.size - 1) % self.maxSteps
    if index == 0 then
      index = self.maxSteps
    end
    local transitionIndex = index % self.maxSteps + 1
    count = count + 1
    indices[count] = index
    transitionIndices[count] = transitionIndex
  end
  local observations = self.observations:index(1, indices)
  local actions = self.actions:index(1, indices)
  local rewards = self.rewards:index(1, indices)
  local terminals = self.terminals:index(1, indices)
  local nextObservations = self.observations:index(1, transitionIndices)
  return {
    observations = observations,
    actions = actions,
    rewards = rewards,
    terminals = terminals,
    nextObservations = nextObservations
  }
end

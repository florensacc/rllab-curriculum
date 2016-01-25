require('torch')
require('nn')
require('nngraph')
require('optim')
--require 'cunn'

require('./cartpole_mdp')

require('./nn_policy')
require('./nn_q_function')
require('./replay_pool')
require('./ou_strategy')
require('./utils')

dbg = require("./debugger")


options = {
  nEpochs = 200,
  batchSize = 32,
  epochLength = 100,
  minPoolSize = 1000,
  evalSamples = 1000,
  replayPoolSize = 1000000,
  discount = 0.99,
  maxPathLength = 100,
  qfWeightDecay = 0.01,
  qfLearningRate = 1e-3,
  policyWeightDecay = 0,
  policyLearningRate = 1e-4,
  softTargetTau = 0.001,
  ouTheta = 0.15,
  ouSigma = 0.3,
  policyH1Size = 32,
  policyH2Size = 32,
  qfH1Size = 32,
  qfH2Size = 32,
}

mdp = CartpoleMDP()

eval_mdp = CartpoleMDP()

policy = NNPolicy(mdp, options.policyH1Size, options.policyH2Size)
targetPolicy = NNPolicy(mdp, options.policyH1Size, options.policyH2Size)
targetPolicy:getParameters():copy(policy:getParameters())

qf = NNQFunction(mdp, options.qfH1Size, options.qfH2Size)
targetQf = NNQFunction(mdp, options.qfH1Size, options.qfH2Size)
targetQf:getParameters():copy(qf:getParameters())

es = OUStrategy.new(mdp, options.ouTheta, options.ouSigma)
pool = ReplayPool(
  mdp:observationShape()[1],
  mdp:actionDim(),
  options.replayPoolSize
)

-- start training

local itr = 0
local pathLength = 0
local pathReturn = 0
local terminal = false
local state, obs = mdp:reset()

local qf_optim_state = {}
local policy_optim_state = {}

for epoch = 1,options.nEpochs do
  print(string.format('epoch %d', epoch))
  for epoch_itr = 1,options.epochLength do
    itr = itr + 1
    if terminal then
      state, obs = mdp:reset()
      es:episodeReset()
      pathLength = 0
      pathReturn = 0
    end
    local action = es:getAction(obs, policy)
    local nextState
    local nextObs
    local reward
    nextState, nextObs, reward, terminal = mdp:step(state, action)
    pathLength = pathLength + 1
    pathReturn = pathReturn + reward

    if pathLength >= options.maxPathLength then
      terminal = true
    end

    pool:addSample(obs, action, reward, terminal)
    state, obs = nextState, nextObs

    if pool.size >= options.minPoolSize then
      batch = pool:randomBatch(options.batchSize)
      observations = batch.observations
      actions = batch.actions
      rewards = batch.rewards
      nextObservations = batch.nextObservations
      terminals = batch.terminals

      -- form the y values
      nextPolicyActions = targetPolicy:getActions(nextObservations)
      nextQVals = targetQf:getQVal(nextObservations, nextPolicyActions)

      ys = rewards + torch.cmul(torch.ones(terminals:size()) - terminals, nextQVals) * options.discount

      -- train the critic
      pred = qf:forward(observations, actions)
      criterion = nn.MSECriterion()
      qf_params, qf_grads = qf:getParameters()
      target_qf_params = targetQf:getParameters()
      policy_params, policy_grads = policy:getParameters()
      target_policy_params = targetPolicy:getParameters()
      local qfeval = function(x)
        if x ~= qf_params then
          qf_params:copy(x)
        end
        qf_grads:zero()
        q_loss = criterion:forward(pred, ys)
        grad_criterion = criterion:backward(pred, ys)
        qf:backward(observations, actions, grad_criterion)
        return q_loss, qf_grads
      end

      optim.adam(qfeval, qf_params, {learningRate = options.qfLearningRate}, qf_optim_state)

      if itr % 100 == 0 then
        print(string.format("Iteration %d; Q loss: %f; Q param norm: %f; target Q param norm: %f", itr, q_loss, torch.norm(qf_params), torch.norm(target_qf_params)))
      end

      -- train the actor
      -- This is a bit tricky
      local policyeval = function(x)
        if x ~= policy_params then
          policy_params:copy(x)
        end
        policy_grads:zero()

        local policyActions = policy:forward(observations)
        local qvals = qf:forward(observations, policyActions)
        local gradAction = qf:backward(observations, policyActions, -torch.ones(options.batchSize) / options.batchSize)[2]

        -- the objective is to maximize Q(s, policy(s)), or equivalently minimize
        -- -Q(s, policy(s)), hence we should pass a vector of -1s as the errors
        policy:backward(observations, gradAction)
        policy_loss = -torch.mean(qvals)

        return policy_loss, policy_grads
      end

      optim.adam(policyeval, policy_params, {learningRate = options.policyLearningRate}, policy_optim_state)

      updateTarget(qf.model, targetQf.model, options.softTargetTau)
      updateTarget(policy.model, targetPolicy.model, options.softTargetTau)
      -- update target networks
      --target_qf_params:mul(1.0 - options.softTargetTau)
      --target_qf_params:add(options.softTargetTau, qf_params)
      --target_policy_params:mul(1.0 - options.softTargetTau)
      --target_policy_params:add(options.softTargetTau, policy_params)

    end -- if pool.size >= options.minPoolSize
  end -- for epoch_itr

  if pool.size >= options.minPoolSize then
    -- evaluate the policy
    n_eval_samples = 0
    eval_path_returns = {}
    while n_eval_samples < options.evalSamples do
      local terminal = false
      local path_return = 0
      local path_length = 0
      state, obs = eval_mdp:reset()
      n_eval_samples = n_eval_samples + 1

      while not terminal and path_length < options.maxPathLength do
        local action = policy:getAction(obs)
        local nextState
        local nextObs
        local reward
        nextState, nextObs, reward, terminal = eval_mdp:step(state, action)
        path_return = path_return + reward
        path_length = path_length + 1
        n_eval_samples = n_eval_samples + 1
        state, obs = nextState, nextObs
      end
      eval_path_returns[#eval_path_returns+1] = path_return
    end
    print(torch.mean(torch.Tensor(eval_path_returns)))
  end -- if pool.size >= options.minPoolSize
end -- for epoch = 1,options.nEpochs

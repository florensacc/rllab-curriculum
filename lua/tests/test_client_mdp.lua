require "../client_mdp"

mdp = ClientMDP("box2d.cartpole_mdp")

print('mdp observation dim:', mdp:observationDim())
print('mdp action dim:', mdp:actionDim())

lb, ub = mdp:actionBounds()

print('action lower bound:', lb)
print('action upper bound:', ub)

print('test reset:', mdp:reset())

state, obs = mdp:reset()
print('reset again')

time = os.time()
for i = 1, 10000 do
  mdp:step(state, torch.zeros(mdp:actionDim()))
end
print(os.difftime(os.time(), time))

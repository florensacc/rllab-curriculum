OUStrategy = require('../ou_strategy')
CartpoleMDP = require('../cartpole_mdp')

mdp = CartpoleMDP()
ou = OUStrategy.new(mdp, 0.15, 0.3)

for i = 1,100 do
  print(ou:_evolveState())
end

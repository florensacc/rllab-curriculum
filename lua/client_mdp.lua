local ClientMDP = torch.class('ClientMDP')


function ClientMDP:__init(mdp_module)
  local zmq = require("lzmq")
  local port
  local f = io.popen(string.format("python scripts/mdp_server.py %s", mdp_module))
  for line in (f:lines()) do
    port = tonumber(line)
    break
  end
  local ctx = zmq.context()
  local socket = ctx:socket{zmq.REQ, connect = string.format("tcp://localhost:%s", port)}
  self.socket = socket
end

--function CartpoleMDP:reset()
--  local ret = self.py.eval('mdp.reset()')
--  return ret[1], ret[2]
--end
--
--function CartpoleMDP:step(state, action)
--  local ret = self.py.eval('mdp.step(state, action)', {state = state, action = action})
--  return ret[1], ret[2], ret[3], ret[4]
--end
--
function ClientMDP:actionDim()
  self.socket:send('action_dim')
  local data, _ = self.socket:recv()
  return tonumber(data)
end

function toArray(str)
  local entries = str:split(',') 
  local val = {}
  for i = 1, #entries do
    val[i] = tonumber(entries[i])
  end
  return val
end

function toArrays(str)
  local arrs = str:split(';') 
  local val = {}
  for i = 1, #arrs do
    val[i] = toArray(arrs[i])
  end
  return val
end


function ClientMDP:observationShape()
  self.socket:send('observation_shape')
  local data, _ = self.socket:recv()
  return toArray(data)
end

function ClientMDP:actionBounds()
  self.socket:send('action_bounds')
  local data, _ = self.socket:recv()
  bounds = toArrays(data)
  return bounds[1], bounds[2]
end

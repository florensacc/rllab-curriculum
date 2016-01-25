local ClientMDP = torch.class('ClientMDP')

function toArray(str)
  local entries = str:split(',') 
  local val = {}
  for i = 1, #entries do
    val[i] = tonumber(entries[i])
  end
  return torch.Tensor(val)
end

function toArrays(str)
  local arrs = str:split(';') 
  local val = {}
  for i = 1, #arrs do
    val[i] = toArray(arrs[i])
  end
  return val
end

function fromArray(arr)
  local arr = arr:view(-1)
  local t = {}
  for i = 1, arr:nElement() do
    t[#t+1] = tostring(arr[i])
  end
  return table.concat(t, ',')
end

function ClientMDP:__init(mdp_module)
  local zmq = require("lzmq")
  local init_port = 4265
  f = io.popen(string.format("python scripts/mdp_server.py %s", mdp_module))
  local ctx = zmq.context()
  local init_socket = ctx:socket{zmq.REQ, connect = string.format("tcp://localhost:%s", init_port)}
  init_socket:send('sup')
  local new_port = init_socket:recv()
  print(new_port)
  init_socket:close()
  local socket = ctx:socket{zmq.REQ, connect = string.format("tcp://localhost:%s", new_port)}
  self.socket = socket
end

function ClientMDP:reset()
  self.socket:send('reset')
  local data, _ = self.socket:recv()
  local parsed = toArrays(data)
  return parsed[1], parsed[2]
end

function ClientMDP:step(state, action)
  local message = string.format(
    'step;%s;%s',
    fromArray(state),
    fromArray(action)
  )
  self.socket:send(message)
  local data, _ = self.socket:recv()
  local arrs = toArrays(data)
  return arrs[1], arrs[2], arrs[3][1], (arrs[4][1] == 1)
end

function ClientMDP:actionDim()
  self.socket:send('action_dim')
  local data, _ = self.socket:recv()
  return tonumber(data)
end

function ClientMDP:observationDim()
  self.socket:send('observation_dim')
  local data, _ = self.socket:recv()
  return tonumber(data)
end

function ClientMDP:actionBounds()
  self.socket:send('action_bounds')
  local data, _ = self.socket:recv()
  bounds = toArrays(data)
  return bounds[1], bounds[2]
end

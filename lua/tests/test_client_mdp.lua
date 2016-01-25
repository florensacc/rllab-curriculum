require "../client_mdp"

mdp = ClientMDP("box2d.cartpole_mdp")

print(mdp:actionBounds())
--f = io.popen("python scripts/mdp_server.py mujoco_1_22.half_cheetah_mdp")
--
--for line in (f:lines()) do
--  port = tonumber(line)
--end
--
--local zmq = require("lzmq")
--
--local ctx = zmq.context()
--local socket = ctx:socket{zmq.REQ, connect = "tcp://localhost:11223"}
--
--socket:send("reset")
----for line in (f:lines()) do
----  port = tonumber(line)
----  break
----end
--
--
--print(socket:recv())

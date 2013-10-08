require 'nngraph'
require 'nndx'

local function t1()
	local d = 5
	local obs = torch.randn(1,d)

	-- does not need to have a dimension, but it must be a tensor...
	local bogus = torch.Tensor()  
	
    -- classical construction (no input needed)
	local ssnet = nndx.Constant({1, 2*d})
   	local distrnet = nn.Linear(2*d, d)
   	local net = nn.Sequential():add(ssnet):add(distrnet)
    
    -- backward pass needs the bogus tensor, forward does not
    print('f1', net:forward())
    print('f1b', net:forward(bogus))
    print('b1b', net:backward(bogus, obs))
    
    -- using source module as input (works, but needs an input)
    local sm = ssnet()
    local gnet = nn.gModule({sm}, {distrnet(sm)})
    
	-- now forward needs the bogus tensor too
    print('f2b', gnet:forward(bogus))
    print('b2b', gnet:backward(bogus, obs))

    -- pure construction, is this a desired way of using graphs? (no inputs: currently not allowed)
    local gnetpure = nn.gModule({}, {distrnet(sm)})
    print('f3', gnetpure:forward())
    print('b3', gnetpure:backward(nil, obs))
end

t1()
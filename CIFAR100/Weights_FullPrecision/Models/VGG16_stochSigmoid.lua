require 'cudnn'
require 'cunn'
require 'nn'
require './unipolarBinarizedNeurons.lua'

numHid = 4096;
local function activation()
	local C = nn.Sequential()
	C:add(nn.Sigmoid())
	C:add(unipolarBinarizedNeurons(opt.stcNeurons))
	return C
end
local model = nn.Sequential()
local p = 0.5

-- block #1
model:add(cudnn.SpatialConvolution(3, 64, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(64, 64, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2))  -- 32 -> 16
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())
--model:add(nn.Dropout(p))

-- block #2
model:add(cudnn.SpatialConvolution(64, 128, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(128, 128, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 16 -> 8
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

-- block #3
model:add(cudnn.SpatialConvolution(128, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 8 -> 4
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

-- block #4

model:add(cudnn.SpatialConvolution(256, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-3))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 4 -> 2
model:add(activation())

-- block #5

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 2 -> 1
model:add(activation())

-- linear layer

model:add(nn.View(512*1*1))

model:add(nn.Linear(512*1*1,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(nn.Linear(numHid,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

--model:add(nn.Dropout(p))

model:add(nn.Linear(numHid,100))

model:add(nn.LogSoftMax())

local dE, param = model:getParameters()
local weight_size = dE:size(1)
local clipvector = torch.Tensor(weight_size):fill(1)

local counter = 0
for i, layer in ipairs(model.modules) do
	if layer.__typename == 'nn.Linear' then
      		local weight_size = layer.weight:size(1)*layer.weight:size(2)
      		local size_w=layer.weight:size();      	
      		clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      	        counter = counter+weight_size
      		local bias_size = layer.bias:size(1)
      		clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      		counter = counter+bias_size
    	elseif layer.__typename == 'cudnn.SpatialBatchNormalization' then
        	local weight_size = layer.weight:size(1)
        	local size_w=layer.weight:size(); 
        	clipvector[{{counter+1, counter+weight_size}}]:fill(0)
        	counter = counter+weight_size
        	local bias_size = layer.bias:size(1)
        	clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        	counter = counter+bias_size
	elseif layer.__typename == 'nn.BatchNormalization' then
      		local weight_size = layer.weight:size(1)
      		clipvector[{{counter+1, counter+weight_size}}]:fill(0)
      		counter = counter+weight_size
      		local bias_size = layer.bias:size(1)
      		clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      		counter = counter+bias_size
	elseif layer.__typename == 'cudnn.SpatialConvolution' then
      		local size_w=layer.weight:size();
      		local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]
      		local filter_size=size_w[3]*size_w[4]
      		clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      		counter = counter+weight_size
      		local bias_size = layer.bias:size(1)
      		clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      		counter = counter+bias_size
	end
end

local function MSRinit(net)
	local function init(name)
		for k,v in pairs(net:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			v.bias:zero()
		end
	end
  	init('cudnn.SpatialConvolution')
	init('nn.Linear')
	
end
function rand_initialize(layer)
  local tn = torch.type(layer)
  if tn == "cudnn.SpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.SpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
 elseif tn == "nn.SpatialConvolutionMM" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "cudnn.VolumetricConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.Linear" then
    local c =  math.sqrt(2.0 / layer.weight:size(2));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.SpatialBachNormalization" then
    layer.weight:fill(1)
    layer.bias:fill(0)
  elseif tn == "cudnn.SpatialBachNormalization" then
    layer.weight:fill(1)
    layer.bias:fill(0)
  end
end

model:apply(rand_initialize)

print(clipvector:ne(0):sum())

return {
	model = model,
	clipV = clipvector,
	}

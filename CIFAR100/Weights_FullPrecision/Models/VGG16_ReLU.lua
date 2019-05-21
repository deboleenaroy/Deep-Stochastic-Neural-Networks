require 'cudnn'
require 'cunn'
require 'nn'

numHid = 4096;
local function activation()
	local C = nn.Sequential()
	C:add(nn.ReLU(true))
	return C
end
local model = nn.Sequential()
local p = 0.3

-- block #1
model:add(cudnn.SpatialConvolution(3, 64, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())
model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(64, 64, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2))  -- 32 -> 16
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())

-- block #2
model:add(cudnn.SpatialConvolution(64, 128, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(128, 128, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 16 -> 8
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

-- block #3
model:add(cudnn.SpatialConvolution(128, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 8 -> 4
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

-- block #4

model:add(cudnn.SpatialConvolution(256, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-3))
model:add(activation())

model:add(nn.Dropout(p))


model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 4 -> 2
model:add(activation())

-- block #5

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(nn.Dropout(p))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 2 -> 1
model:add(activation())

-- linear layer

model:add(nn.View(512*1*1))

model:add(nn.Linear(512*1*1,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

model:add(nn.Dropout(p))

model:add(nn.Linear(numHid,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

model:add(nn.Dropout(p))

model:add(nn.Linear(numHid,100))

model:add(nn.LogSoftMax())

-- initialization from MSR
local function MSRinit(net)
	local function init(name)
		for k,v in pairs(net:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			v.bias:zero()
		end
	end
  	init'cudnn.SpatialConvolution'
end

MSRinit(model)

return {
	model = model
	}

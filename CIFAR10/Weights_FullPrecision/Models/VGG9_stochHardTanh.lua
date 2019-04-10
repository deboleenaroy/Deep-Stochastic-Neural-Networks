--[[This code specify the model for CIFAR 10 dataset. This model uses the Shift based batch-normalization algorithm.
In this file we also secify the Glorot learning parameter and the which of the learnable parameter we clip ]]
require 'cudnn'
require 'cunn'
require 'nn'
require './bipolarBinarizedNeurons.lua'

numHid=1024;
local model = nn.Sequential()

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 128, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(128, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(cudnn.SpatialConvolution(128, 128, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2))

model:add(cudnn.SpatialBatchNormalization(128, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(cudnn.SpatialConvolution(128, 256, 3, 3 ,1,1,1,1))
model:add(cudnn.SpatialBatchNormalization(256, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3 ,1,1,1,1))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(cudnn.SpatialBatchNormalization(256, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(cudnn.SpatialConvolution(256, 512, 3, 3,1,1,1,1))
model:add(cudnn.SpatialBatchNormalization(512, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(cudnn.SpatialConvolution(512, 512, 3, 3,1,1,1,1))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(cudnn.SpatialBatchNormalization(512, 1e-3))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(nn.View(512*4*4))

model:add(nn.Linear(512*4*4,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))

model:add(nn.Linear(numHid,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(nn.HardTanh())
model:add(bipolarBinarizedNeurons(opt.stcNeurons))


model:add(nn.Linear(numHid,10))

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
     model = model,
     --lrs = learningRates,
     --clipV =clipvector,
  }

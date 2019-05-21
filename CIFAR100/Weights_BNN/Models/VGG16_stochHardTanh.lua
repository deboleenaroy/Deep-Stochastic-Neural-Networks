require 'cudnn'
require 'cunn'
require 'nn'
require './BinaryLinear'
require './bipolarBinarizedNeurons.lua'
require './cudnnBinarySpatialConvolution.lua'


numHid = 4096;
local function activation()
	local C = nn.Sequential()
	C:add(nn.HardTanh())
	C:add(bipolarBinarizedNeurons(opt.stcNeurons))
	return C
end
local model = nn.Sequential()

-- block #1
model:add(cudnnBinarySpatialConvolution(3, 64, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(64, 64, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2))  -- 32 -> 16
model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
model:add(activation())

-- block #2
model:add(cudnnBinarySpatialConvolution(64, 128, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(128, 128, 3, 3,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 16 -> 8
model:add(cudnn.SpatialBatchNormalization(128, 1e-5))
model:add(activation())

-- block #3
model:add(cudnnBinarySpatialConvolution(128, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(256, 256, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 8 -> 4
model:add(cudnn.SpatialBatchNormalization(256, 1e-5))
model:add(activation())

-- block #4

model:add(cudnnBinarySpatialConvolution(256, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-3))
model:add(activation())


model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 4 -> 2
model:add(activation())

-- block #5

model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(activation())

model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3 ,1,1,1,1 ))
model:add(cudnn.SpatialBatchNormalization(512, 1e-5))
model:add(cudnn.SpatialMaxPooling(2, 2)) -- 2 -> 1
model:add(activation())

-- linear layer

model:add(nn.View(512*1*1))

model:add(BinaryLinear(512*1*1,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

model:add(BinaryLinear(numHid,numHid))
model:add(nn.BatchNormalization(numHid))
model:add(activation())

model:add(BinaryLinear(numHid,100))
model:add(nn.BatchNormalization(100))

model:add(nn.LogSoftMax())

local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local clipvector = torch.Tensor(weight_size):fill(1)
local counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'BinaryLinear' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
    elseif layer.__typename == 'BatchNormalizationShiftPow2' then
        local weight_size = layer.weight:size(1)
        local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
        learningRates[{{counter+1, counter+weight_size}}]:fill(1)
        clipvector[{{counter+1, counter+weight_size}}]:fill(0)
        counter = counter+weight_size
        local bias_size = layer.bias:size(1)
        learningRates[{{counter+1, counter+bias_size}}]:fill(1)
        clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        counter = counter+bias_size
    elseif layer.__typename == 'nn.BatchNormalization' then
      local weight_size = layer.weight:size(1)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      clipvector[{{counter+1, counter+weight_size}}]:fill(0)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
    elseif layer.__typename == 'SpatialBatchNormalizationShiftPow2' then
        local weight_size = layer.weight:size(1)
        local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
        learningRates[{{counter+1, counter+weight_size}}]:fill(1)
        clipvector[{{counter+1, counter+weight_size}}]:fill(0)
        counter = counter+weight_size
        local bias_size = layer.bias:size(1)
        learningRates[{{counter+1, counter+bias_size}}]:fill(1)
        clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        counter = counter+bias_size
    elseif layer.__typename == 'nn.SpatialBatchNormalization' then
            local weight_size = layer.weight:size(1)
            local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
            learningRates[{{counter+1, counter+weight_size}}]:fill(1)
            clipvector[{{counter+1, counter+weight_size}}]:fill(0)
            counter = counter+weight_size
            local bias_size = layer.bias:size(1)
            learningRates[{{counter+1, counter+bias_size}}]:fill(1)
            clipvector[{{counter+1, counter+bias_size}}]:fill(0)
            counter = counter+bias_size
    elseif layer.__typename == 'cudnnBinarySpatialConvolution' then
      local size_w=layer.weight:size();
      local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]

      local filter_size=size_w[3]*size_w[4]
      GLR=1/torch.sqrt(1.5/(size_w[1]*filter_size+size_w[2]*filter_size))
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
      elseif layer.__typename == 'BinarySpatialConvolution' then
        local size_w=layer.weight:size();
        local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]

        local filter_size=size_w[3]*size_w[4]
        GLR=1/torch.sqrt(1.5/(size_w[1]*filter_size+size_w[2]*filter_size))
        GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
        learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
        clipvector[{{counter+1, counter+weight_size}}]:fill(1)
        counter = counter+weight_size
        local bias_size = layer.bias:size(1)
        learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
        clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        counter = counter+bias_size

  end
end

print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(clipvector:ne(0):sum())
print(counter)

return  {
	model = model,
	lrs = learningRates,
	clipV =clipvector,
	}

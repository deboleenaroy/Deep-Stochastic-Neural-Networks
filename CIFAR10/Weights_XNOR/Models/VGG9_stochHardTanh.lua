--[[This code specify the model for CIFAR 10 dataset. This model uses the Shift based batch-normalization algorithm.
In this file we also secify the Glorot learning parameter and the which of the learnable parameter we clip ]]

function createModel()

	require 'nn'
	require 'cunn'
	require 'cudnn'
	require './bipolarBinarizedNeurons.lua'
	local function activation()
		local C = nn.Sequential()
		C:add(nn.HardTanh())
		C:add(bipolarBinarizedNeurons(opt.stcNeurons))
		return C
	end

	local function MaxPooling(kW, kH, dW, dH, padW, padH)
    		return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   	end

   	local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        	local C= nn.Sequential()
          	C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          	C:add(activation())
   		C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
   		return C
   	end

    	local function BinMaxConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,mW,mH)
        	local C= nn.Sequential()
          	C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          	C:add(activation())
          	C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
          	C:add(MaxPooling(2,2))
       		return C
  	end

	numHid=1024;
	local features = nn.Sequential()
	-- Convolution Layers

	features:add(cudnn.SpatialConvolution(3, 128, 3, 3 , 1, 1, 1, 1))
        features:add(nn.SpatialBatchNormalization(128, 1e-5))
        features:add(cudnn.ReLU(true))	
	        
	features:add(BinMaxConvolution(128, 128, 3, 3, 1, 1, 1, 1))  -- BNORM-> activation -> Conv -> MaxPool (32->16)

	features:add(BinConvolution(128, 256, 3, 3 , 1, 1, 1, 1)) -- BNORM-> activation -> Conv 

	features:add(BinMaxConvolution(256, 256, 3, 3 , 1, 1, 1, 1)) -- BNORM-> activation -> Conv -> MaxPool (16 -> 8)

	features:add(BinConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- BNORM-> activation -> Conv
		
	features:add(BinMaxConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- BNORM-> activation -> Conv -> MaxPool
		
	features:add(BinConvolution(512, numHid, 4, 4, 1, 1, 0, 0)) -- BNORM-> activation -> Conv
		
	features:add(BinConvolution(numHid, numHid, 1, 1, 1, 1, 0, 0)) -- BNORM-> activation -> Conv

	features:add(nn.SpatialBatchNormalization(numHid, 1e-4, false))

	features:add(nn.ReLU(true))

	features:add(cudnn.SpatialConvolution(numHid, 10, 1, 1, 1, 1, 0, 0))

	features:add(nn.View(10))

	features:add(nn.LogSoftMax())

	local model = features
	return model
end

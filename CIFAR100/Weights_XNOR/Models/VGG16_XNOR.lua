
function createModel()

	require 'cudnn'
	require 'cunn'
	require 'nn'
        
	local function activation()
		local C = nn.Sequential()
		C:add(nn.BinActiveZ())
		return C
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
       		C:add(nn.SpatialMaxPooling(2,2))
		return C
	end

	numHid = 4096
	local model = nn.Sequential()
	local p = 0.5

	-- block #1
	model:add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1 ))
	model:add(cudnn.SpatialBatchNormalization(64, 1e-5))
	model:add(nn.ReLU(true))

	model:add(BinMaxConvolution(64, 64, 3, 3, 1, 1, 1, 1 )) -- BNORM-> activation -> Conv -> MaxPool (32->16)

	
	-- block #2
	model:add(BinConvolution(64, 128, 3, 3, 1, 1, 1, 1 )) -- BNORM-> activation -> Conv
	
	model:add(BinMaxConvolution(128, 128, 3, 3, 1, 1, 1, 1 )) -- BNORM-> activation -> Conv -> MaxPool (16->8)
	
	-- block #3
	model:add(BinConvolution(128, 256, 3, 3 ,1,1,1,1 ))
	
	model:add(BinConvolution(256, 256, 3, 3 ,1,1,1,1 ))
	
	model:add(BinMaxConvolution(256, 256, 3, 3 ,1,1,1,1 ))
	
	-- block #4
	
	model:add(BinConvolution(256, 512, 3, 3 ,1,1,1,1 ))
	
	model:add(BinConvolution(512, 512, 3, 3, 1, 1, 1, 1 ))
	
	model:add(BinMaxConvolution(512, 512, 3, 3, 1, 1, 1, 1 ))
	
	-- block #5
	
	model:add(BinConvolution(512, 512, 3, 3, 1, 1, 1, 1 ))
	
	model:add(BinConvolution(512, 512, 3, 3, 1, 1, 1, 1 ))
	
	model:add(BinMaxConvolution(512, 512, 3, 3, 1, 1, 1, 1 ))
	
	-- linear layer
	
	model:add(BinConvolution(512, numHid, 1, 1, 1, 1, 0, 0))
	
	model:add(BinConvolution(numHid, numHid, 1, 1, 1, 1, 0, 0))
	
	model:add(nn.SpatialBatchNormalization(numHid, 1e-4, false))
	
	model:add(nn.ReLU(true))
	
	model:add(cudnn.SpatialConvolution(numHid, 100, 1, 1, 1, 1, 0, 0))
	
	model:add(nn.View(100))
	
	model:add(nn.LogSoftMax())
	
        return model

end


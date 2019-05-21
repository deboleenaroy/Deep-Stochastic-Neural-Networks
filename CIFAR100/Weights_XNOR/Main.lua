--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'newLayers'
require 'gnuplot'
require 'pl'
require 'trepl'
cmd = torch.CmdLine()
cmd:addTime()
torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

opt.imageSize = 32
opt.imageCrop = 32

opt.dataset = 'cifar100'
opt.data = '/data/roy77/stochastic_NN/XNOR_Dataset/cifar100/dataset'
opt.nClasses = 100
opt.nEpochs = 400
opt.epochSize = 1000
opt.batchSize = 50
opt.epochNumber = 1
opt.binaryWeight = true
opt.stcNeurons = false
opt.testOnly = false
opt.netType = 'ResNet20_stochHardTanh_withReLU'
opt.optimType = 'sgd'
opt.save = './Results/ResNet20_detHardTanh_withReLU_typeB_v2/'
opt.shortcutType = 'B'

print(opt)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')

torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

cmd:log(opt.save .. '/Log.txt', opt)
confusion = optim.ConfusionMatrix(classes)
epoch = opt.epochNumber
if opt.testOnly then
	test()
else
  best_acc = 0
  local errorLogFilename = paths.concat(opt.save,'ErrorRate.log')
  local lossLogFilename = paths.concat(opt.save,'Loss.log')
  local errorLog = optim.Logger(errorLogFilename)
  local lossLog = optim.Logger(lossLogFilename)
  for i=1,opt.nEpochs do
        print('Epoch ' .. epoch)
	--print('-------------------LR-------------------')
        --print(optimState.learningRate)
	confusion:zero()
   	AccTrain, LossTrain = train()
        confusion:updateValids()
	ErrTrain = (1-confusion.totalValid)
	if #classes <= 10 then
        	print(confusion)
        end

	print('Training Error = ' .. ErrTrain)
        print('Training Loss = ' .. LossTrain) 

	confusion:zero()
   	AccTest, LossTest = test()
	confusion:updateValids()
	ErrTest = (1-confusion.totalValid)
	if #classes <= 10 then
        	print(confusion)
        end
	print('Test Error = ' .. ErrTest)
        print('Test Loss = ' .. LossTest) 


        errorLog:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    	lossLog:add{['Training Loss']= LossTrain, ['Test Loss'] = LossTest}
	errorLog:style{['Training Error'] = '-', ['Test Error'] = '-'}
        errorLog:plot()
        lossLog:style{['Training Loss'] = '-', ['Test Loss'] = '-'}
        lossLog:plot()
       	
   	if AccTest > best_acc then
   		saveDataParallel(paths.concat(opt.save, 'best_model.t7'), model) -- defined in util.lua
   		--torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
		best_acc = AccTest
   	end
   	epoch = epoch + 1
	print('Best Accuracy = ' .. best_acc)
  end
end

local unipolarBinarizedNeurons,parent = torch.class('unipolarBinarizedNeurons', 'nn.Module')


function unipolarBinarizedNeurons:__init(stcFlag)
   parent.__init(self)
   self.stcFlag = stcFlag
   self.randmat=torch.Tensor();
   self.outputR=torch.Tensor();
end
function unipolarBinarizedNeurons:updateOutput(input)
    self.randmat:resizeAs(input);
    self.outputR:resizeAs(input);
    self.output:resizeAs(input);
    self.outputR:copy(input)
    --self.output = torch.lt(r,p):cuda()
    if self.stcFlag then
    	local mask=self.outputR-self.randmat:rand(self.randmat:size())
    	self.output=mask:sign()
    	self.output:add(1):div(2)
    else 
    	self.output = self.outputR:add(-0.5):sign()
	self.output:add(1):div(2)
    end
    return self.output
end

function unipolarBinarizedNeurons:updateGradInput(input, gradOutput)
        self.gradInput:resizeAs(gradOutput)
        self.gradInput:copy(gradOutput) 
   return self.gradInput
end

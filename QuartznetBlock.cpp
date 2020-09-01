#include "QuartnetBlock.h"



int computeNewKernelSize(int kernelSize, double kernelWidth){
	int newKernelSize = std::max((int)(kernelSize * kernelWidth), 1);
	if (newKernelSize % 2 == 0){
		newKernelSize += 1;
	}
	return newKernelSize;
}


int getSamePadding(int kernelSize, int stride, int dilation){
	if (stride > 1 && dilation > 1){
		throw std::invalid_argument("Only stride OR dilation may be greater than 1");
	}
	if (dilation > 1){
		return ((dilation * kernelSize) / 2) -1;
	}
	return kernelSize / 2;
}


//using namespace fl;
QuartznetBlock::QuartznetBlock(
	  int inChannels,
	  int outChannels,
	  int repeat,
	  int kernelSize,
	  int stride,
	  int dilation,
	  double dropOut,
	  bool residual,
	  bool separable,
	  std::string residualMode,
	  bool lNormIncludeTime){

	this->residual = residual;
	double kernelSizeFactor = 1.0; // fix kernel size factor.

	int newKernelSize = computeNewKernelSize(kernelSize, kernelSizeFactor);
	std::cout << "gia tri new kernel size: " << newKernelSize << std::endl;


	int paddingVal = getSamePadding(kernelSize, stride, dilation);
	std::cout << "gia tri padding: " << paddingVal << std::endl;
	fl::Sequential convList;

	int inChannelsLoop = inChannels;
	for (int i=0; i<(repeat - 1); i++){
		auto listConvsFirst = this->getConvNormLayer(inChannelsLoop,
												 	 outChannels,
												 	 newKernelSize,
													 stride,
													 dilation,
													 paddingVal,
													 separable,
													 lNormIncludeTime);
		convList.add(listConvsFirst);
		auto layerActDrop = this->getActivationAndDropoutLayer(dropOut);
		convList.add(layerActDrop);
		inChannelsLoop = outChannels;
	}

	auto lastConvLayer = this->getConvNormLayer(inChannelsLoop,
												outChannels,
												newKernelSize,
												stride,
												dilation,
												paddingVal,
												separable,
												lNormIncludeTime);

	convList.add(lastConvLayer);

	add(convList);

	if (residual){

		if (residualMode.compare("stride_add")){
			stride = stride;
		} else {
			stride = 1;
		}

		auto layerConvResidual = this->getConvNormLayer(
				inChannels,
				outChannels,
				1,		// kernelSize = 1
				stride,
				1,
				0,
				false, // separable = false
				lNormIncludeTime);
		std::cout << "thong so residual"<< inChannels << "-"<< outChannels << "-"<< stride << std::endl;
		add(layerConvResidual);
	}

	auto lastDropoutLayer = this->getActivationAndDropoutLayer(dropOut);
	add(lastDropoutLayer);

}


fl::Sequential QuartznetBlock::getConvNormLayer(
		int inChannels,
		int outChannels,
		int kernelSize,
		int stride,
		int dilation,
		int padding,
		bool separable,
		bool normIncludeTime){

	fl::Sequential mListConvs;

	if (separable){
		auto depthWiseConv2d = fl::Conv2D(
				inChannels,
				inChannels,
				kernelSize,
				1,
				stride,
				1,
				padding,
				0,
				dilation,
				1,
				false, // bias
				1); // groups

		auto pointWiseConv2d = fl::Conv2D(
				inChannels,
				outChannels,
				1,	// kernel size x
				1,  // kernel size y
				stride,
				1,
				0,
				0,
				dilation,
				1,
				false, // bias
				1); //groups
		mListConvs.add(depthWiseConv2d);
		mListConvs.add(pointWiseConv2d);

	} else {
		auto normalConv2d = fl::Conv2D(
				inChannels,
				outChannels,
				kernelSize,
				1,
				stride,
				1,
				padding,
				0,
				dilation,
				1,
				false,
				1);

		mListConvs.add(normalConv2d);
	}

	if (normIncludeTime){
		auto layerNormWithTime = fl::LayerNorm(std::vector<int>{0, 1, 2});
		mListConvs.add(layerNormWithTime);
	} else {
		auto layerNormWithoutTime = fl::LayerNorm(std::vector<int>{1, 2});
		mListConvs.add(layerNormWithoutTime);
	}

	// need implement group shuffle later.

	return mListConvs;
}

fl::Sequential QuartznetBlock::getActivationAndDropoutLayer(double dropOut){
	fl::Sequential listActDrop;

	auto layerActivation = fl::ReLU();
	auto layerDropOut = fl::Dropout(dropOut);

	listActDrop.add(layerActivation);
	listActDrop.add(layerDropOut);

	return listActDrop;
}


std::vector<fl::Variable> QuartznetBlock::forward(const std::vector<fl::Variable>& inputs) {
  std::cout << "so module: " << this->modules().size() << std::endl;
  auto out = inputs[0];
  std::cout << "shape input: " << out.dims() << std::endl;

  auto outputConv = module(0)->forward({out})[0];
  std::cout << "shape after component outputConv: " << outputConv.dims() << std::endl;

  auto outputAfterConv = outputConv;
  if (this->residual){
	  std::cout << "shape input before residual: "<< out.dims()<< std::endl;
	  auto outputResidual = module(1)->forward({out})[0];
	  std::cout << "shape after residual: " << outputResidual.dims() << std::endl;
	  outputAfterConv = outputAfterConv + outputResidual;
	  std::cout << "shape after cat with residual: " << outputAfterConv.dims() << std::endl;
	  auto outputFinal = module(2)->forward({outputAfterConv});
	  return outputFinal;
  }

  auto outputFinal = module(1)->forward({outputAfterConv});
  return outputFinal;
}


std::string QuartznetBlock::prettyString() const {
  std::ostringstream ss;

  return ss.str();
}

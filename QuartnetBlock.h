//#pragma once

#include <flashlight/flashlight.h>
#include <string>
#include <vector>
#include <iostream>

//namespace w2l {


class QuartznetBlock : public fl::Container {
 	public:
	bool residual = true;
//	fl::Sequential convList;


	explicit QuartznetBlock(
	  int inChannels,
	  int outChannels,
	  int repeat,
	  int kernelSize,
	  int stride,
	  int dilation,
	  double dropout,
	  bool residual,
	  bool separabel,
	  std::string residualMode,
	  bool lNormIncludeTime);

	std::vector<fl::Variable> forward(const std::vector<fl::Variable>& inputs) override;
	std::string prettyString() const override;
	fl::Sequential getConvNormLayer(
									int inChannels,
									int outChannels,
									int kernelSize,
									int stride,
									int dilation,
									int padding,
									bool separable,
									bool normIncludeTime);
	fl::Sequential getActivationAndDropoutLayer(double dropOut);
//  fl::Module getConvLayer();

};

//}

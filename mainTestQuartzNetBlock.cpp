/*
 * mainTestQuartzNetBlock.cpp
 *
 *  Created on: Aug 27, 2020
 *      Author: trangtv
 */

#include <iostream>
#include "QuartnetBlock.h"


using namespace std;

int main() {

//	af::array inputX = af::randn(af::dim4(76, 80, 100, 1));
//	std::vector<af::array> fields{inputX};
//	auto ds = std::make_shared<fl::TensorDataset>(fields);
//	fl::BatchDataset trainset(ds, 8);
//
////	std::vector<af::array> inputs;

//	auto tensor = af::randn(76, 80, 1, 12);
//	auto tensor = af::randn(af::dim4(76, 80, 1, 12));
	auto tensor = af::randn(af::dim4(76, 64, 1, 2));
//	af::print("origin tensor: ", fl::Variable(tensor, false).array());
	std::vector<af::array> fields{tensor};
	auto ds = std::make_shared<fl::TensorDataset>(fields);
	fl::BatchDataset trainset(ds, 8);

	int inChannels = 1;
	int outputChannels = 4;
	int repeat = 1;
	int kernelSize = 33;
	int stride = 2;
	int dilation = 1;
	double dropOut = 0.1;
	bool residual = false;
	bool separable = true;
	std::string residualMode = "add";
	bool lNormIncludeTime = false;

	fl::Sequential scratchNemo;
	auto quartznetBlock1 = QuartznetBlock(
				1, //inChannels,
				4, //outputChannels,
				1, //repeat,
				33, //kernelSize,
				2, //stride,
				1, //dilation,
				0.1, //dropOut,
				false, //residual,
				true, //separable,
				"add", //residualMode,
				false //lNormIncludeTime
			);

	auto quartznetBlock2 = QuartznetBlock( 4, 4, 5, 33, 1, 1, 0.1, true, true, "add", false);
	auto quartznetBlock3 = QuartznetBlock( 4, 8, 5, 51, 1, 1, 0.1, true, true, "add", false);
	auto quartznetBlock4 = QuartznetBlock( 8, 8, 5, 75, 1, 1, 0.1, true, true, "add", false);
	auto quartznetBlock5 = QuartznetBlock( 8, 8, 1, 87, 1, 1, 0.1, false, true, "add", false);
	auto quartznetBlock6 = QuartznetBlock( 8, 16, 1, 1, 1, 1, 0.1, false, false, "add", false);

	scratchNemo.add(quartznetBlock1);
	scratchNemo.add(quartznetBlock2);
	scratchNemo.add(quartznetBlock3);
	scratchNemo.add(quartznetBlock4);
	scratchNemo.add(quartznetBlock5);
	scratchNemo.add(quartznetBlock6);

	auto tVariable = fl::Variable(tensor, false);
	auto result = scratchNemo(tVariable);
	cout << result.dims() << endl;
//	auto result_af1 = quartznetBlock1({tVariable})[0];
//	auto result_af2 = quartznetBlock2({result_af1})[0];
//	cout << "result af2" << endl;
//	auto result_af3 = quartznetBlock3({result_af2})[0];
//	cout << "result af 3" << endl;

//	auto layerReorder = fl::Reorder(0, 1, 3, 2);
//	for (auto &batch : trainset) {
//		auto eBatchOrigin = batch[0];
////		cout << eBatchOrigin.dims() << endl;
//
////		eBatchOrigin = af::reorder(eBatchOrigin, 0, 1, 3, 2);
////		cout << "shape of each batch" << eBatchOrigin.dims() << endl;
//		auto outputBatchBeforeForward = layerReorder(fl::Variable(batch[0], false));
//		auto result = quartznetBlockFirst({outputBatchBeforeForward});
//		cout << result[0].dims() << endl;
////		cout << outputBatchBeforeForward.dims() <<endl;
//	}

}


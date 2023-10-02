package main

import (
	"fmt"

	"github.com/codehex/neuralnet"
)

func main() {
	// Generate hyperparameters
	hyperParams, err := neuralnet.NewHyperParametersBuilder().
		AddLayers(neuralnet.ActivationFuncNameReLU, 6, 6).
		AddLayers(neuralnet.ActivationFuncNameSigmoid, 1).
		SetLearningRate(0.02).
		SetIterations(2000).
		SetRegularizationFactor(0.5).
		SetDropoutKeepProbability(0.75).
		SetMiniBatchSize(1024).
		Build()

	if err != nil {
		panic(err)
	}
	fmt.Println(hyperParams)

	// Read training data model
	trainingDataSet, err := neuralnet.NewImageSetBuilder().
		AugmentFlipHorizontal().
		WithPathPrefix("../datasets/Vegetable Images/train").
		AddFolder("Cabbage", false).
		AddFolder("Carrot", true).
		ResizeImages(64, 64).
		Normalize().
		Build()

	if err != nil {
		panic(err)
	}

	// Use hyperparameters to train model
	model, err := hyperParams.TrainModel(trainingDataSet)

	if err != nil {
		panic(err)
	}

	model.Predict(trainingDataSet)

	// Read test data
	testDataSet, err := neuralnet.NewImageSetBuilder().
		WithPathPrefix("../datasets/Vegetable Images/test").
		AddFolder("Cabbage", false).
		AddFolder("Carrot", true).
		ResizeImages(64, 64).
		Normalize().
		Build()

	if err != nil {
		panic(err)
	}

	// Use model to generate predictions
	model.Predict(testDataSet)
}

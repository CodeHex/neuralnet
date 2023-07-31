package main

import (
	"fmt"

	"github.com/codehex/neuralnet"
)

func main() {
	// Generate hyperparameters
	hyperParams, err := neuralnet.NewHyperParametersBuilder().
		AddLayers(neuralnet.ActivationFuncNameReLU, 3, 2).
		AddLayers(neuralnet.ActivationFuncNameSigmoid, 1).
		SetInitFactor(0.01).
		SetLearningRate(0.15).
		SetIterations(5000).
		Build()

	if err != nil {
		panic(err)
	}
	fmt.Println(hyperParams)

	// Read training data model
	trainingDataSet, err := neuralnet.NewImageSetBuilder().
		WithPathPrefix("../datasets/Vegetable Images/train").
		AddFolder("Cabbage", false).
		AddFolder("Carrot", true).
		ResizeImages(32, 32).
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
		ResizeImages(32, 32).
		Build()

	if err != nil {
		panic(err)
	}

	// Use model to generate predictions
	model.Predict(testDataSet)
}

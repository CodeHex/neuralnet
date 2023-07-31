package main

import "fmt"

func main() {
	// Generate hyperparameters
	hyperParams, err := NewHyperParametersBuilder().
		AddMultipleLayers("relu", 2, 5, 4).
		AddLayer("sigmoid", 1).
		SetLearningRate(0.01).
		SetIterations(2000).
		Build()

	if err != nil {
		panic(err)
	}
	fmt.Println(hyperParams)

	// Read training data model
	trainingDataSet, err := NewImageSetBuilder().
		WithLogging().
		WithPathPrefix("datasets/Vegetable Images/train").
		AddFolder("Cabbage", false).
		AddFolder("Carrot", true).
		ResizeImages(64, 64).
		Build()

	if err != nil {
		panic(err)
	}

	// Use hyperparameters to train model
	_, err = hyperParams.TrainModel(trainingDataSet)

	if err != nil {
		panic(err)
	}

	// Read test data

	// Use model to generate predictions
}

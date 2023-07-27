package main

import "fmt"

type LayerParameters struct {
	layer   LayerDefinition
	weights [][]float64
	biases  []float64
}

type Model struct {
	hyperParams HyperParameters
	params      []LayerParameters
}

func NewHyperParametersBuilder() HyperParametersBuilder {
	return HyperParametersBuilder{}
}

func main() {
	// Generate hyperparameters
	hyperParams, err := NewHyperParametersBuilder().
		AddMultipleLayers("relu", 2, 5, 4).
		AddLayer("sigmoid", 1).
		SetLearningRate(0.01).
		Build()

	if err != nil {
		panic(err)
	}
	fmt.Println(hyperParams)

	// Read training data model
	_, err = NewImageSetBuilder().
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

	// Read test data

	// Use model to generate predictions
}

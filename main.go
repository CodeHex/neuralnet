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

	// Use hyperparameters to train model

	// Use model to generate predictions
}

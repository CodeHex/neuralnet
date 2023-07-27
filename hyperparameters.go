package main

import (
	"errors"
	"fmt"
)

type LayerDefinition struct {
	neurons            uint
	activationFunction string
}

type HyperParameters struct {
	layers       []LayerDefinition
	learningRate float64
}

type HyperParametersBuilder struct {
	params HyperParameters
}

func (builder HyperParametersBuilder) AddLayer(activationFunction string, neurons uint) HyperParametersBuilder {
	builder.params.layers = append(builder.params.layers, LayerDefinition{neurons, activationFunction})
	return builder
}

func (builder HyperParametersBuilder) AddNLayers(activationFunction string, neurons uint, n uint) HyperParametersBuilder {
	for i := uint(0); i < n; i++ {
		builder.params.layers = append(builder.params.layers, LayerDefinition{neurons, activationFunction})
	}
	return builder
}

func (builder HyperParametersBuilder) AddMultipleLayers(activationFunction string, neurons ...uint) HyperParametersBuilder {
	for _, n := range neurons {
		builder.params.layers = append(builder.params.layers, LayerDefinition{n, activationFunction})
	}
	return builder
}

func (builder HyperParametersBuilder) SetLearningRate(learningRate float64) HyperParametersBuilder {
	builder.params.learningRate = learningRate
	return builder
}

func (builder HyperParametersBuilder) Build() (HyperParameters, error) {
	for _, layer := range builder.params.layers {
		if layer.neurons == 0 {
			return HyperParameters{}, errors.New("layer with 0 neurons defined")
		}
	}

	if builder.params.learningRate <= 0 {
		return HyperParameters{}, errors.New("learning rate must be greater than 0")
	}

	if len(builder.params.layers) == 0 {
		return HyperParameters{}, errors.New("no layers defined")
	}
	return builder.params, nil
}

func (h HyperParameters) String() string {
	title := fmt.Sprintf("number of layers: %d, learning rate: %.3f", len(h.layers), h.learningRate)
	layers := "layers:\n"
	for i := range h.layers {
		layers += fmt.Sprintf("  layer %d - %d neuron(s), %s activation function\n", i+1, h.layers[i].neurons, h.layers[i].activationFunction)
	}
	return "Hyperparameters:\n" + title + "\n" + layers
}

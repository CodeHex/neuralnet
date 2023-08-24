package neuralnet

import (
	"errors"
	"fmt"
	"math"
)

type ActivationFuncName string

const (
	ActivationFuncNameReLU    ActivationFuncName = "relu"
	ActivationFuncNameSigmoid ActivationFuncName = "sigmoid"
	ActivationFuncNameTanh    ActivationFuncName = "tanh"
)

type layerDefinition struct {
	neurons           uint
	actFuncLabel      ActivationFuncName
	activationFunc    func(float64) float64
	activationDerFunc func(float64) float64
}

type HyperParameters struct {
	layers       []layerDefinition
	learningRate float64
	iterations   uint
	initFactor   float64
}

type HyperParametersBuilder struct {
	params HyperParameters
}

func NewHyperParametersBuilder() HyperParametersBuilder {
	return HyperParametersBuilder{
		params: HyperParameters{
			learningRate: 0.01,
			iterations:   1000,
			initFactor:   0.01,
		},
	}
}

func (builder HyperParametersBuilder) AddLayers(a ActivationFuncName, neurons ...uint) HyperParametersBuilder {
	for _, n := range neurons {
		builder.params.layers = append(builder.params.layers, layerDefinition{n, a, getActivationFunc(a), getActivationDerFunc(a)})
	}
	return builder
}

func (builder HyperParametersBuilder) AddNLayers(a ActivationFuncName, neurons uint, n uint) HyperParametersBuilder {
	for i := uint(0); i < n; i++ {
		builder.params.layers = append(builder.params.layers, layerDefinition{neurons, a, getActivationFunc(a), getActivationDerFunc(a)})
	}
	return builder
}

func (builder HyperParametersBuilder) SetLearningRate(learningRate float64) HyperParametersBuilder {
	builder.params.learningRate = learningRate
	return builder
}

func (builder HyperParametersBuilder) SetIterations(iterations uint) HyperParametersBuilder {
	builder.params.iterations = iterations
	return builder
}

func (builder HyperParametersBuilder) SetInitFactor(initFactor float64) HyperParametersBuilder {
	builder.params.initFactor = initFactor
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

	if builder.params.iterations == 0 {
		return HyperParameters{}, errors.New("number of iterations must be greater than 0")
	}

	if len(builder.params.layers) == 0 {
		return HyperParameters{}, errors.New("no layers defined")
	}

	if builder.params.initFactor <= 0 {
		return HyperParameters{}, errors.New("initialization factor must be greater than 0")
	}

	if builder.params.layers[len(builder.params.layers)-1].actFuncLabel != ActivationFuncNameSigmoid {
		return HyperParameters{}, errors.New("last layer must have sigmoid activation function")
	}

	if builder.params.layers[len(builder.params.layers)-1].neurons != 1 {
		return HyperParameters{}, errors.New("last layer must have 1 neuron")
	}
	return builder.params, nil
}

func (h HyperParameters) String() string {
	title := fmt.Sprintf("number of layers: %d, learning rate: %.3f, iterations: %d, init factor: %.2f",
		len(h.layers), h.learningRate, h.iterations, h.initFactor)
	layers := "layers:\n"
	for i := range h.layers {
		layers += fmt.Sprintf("  layer %d - %d neuron(s), %s activation function\n", i+1, h.layers[i].neurons, h.layers[i].actFuncLabel)
	}
	return "Hyperparameters:\n" + title + "\n" + layers
}

func (h HyperParameters) Layer(i int) layerDefinition {
	return h.layers[i-1]
}

func (l layerDefinition) ActivationFunc() func(float64) float64 {
	return l.activationFunc
}

func (l layerDefinition) ActivationDerivativeFunc() func(float64) float64 {
	return l.activationDerFunc
}

func getActivationFunc(a ActivationFuncName) func(float64) float64 {
	switch a {
	case ActivationFuncNameReLU:
		return relu
	case ActivationFuncNameSigmoid:
		return sigmoid
	case ActivationFuncNameTanh:
		return tanh
	default:
		return relu
	}
}

func getActivationDerFunc(a ActivationFuncName) func(float64) float64 {
	switch a {
	case ActivationFuncNameReLU:
		return reluDerivative
	case ActivationFuncNameSigmoid:
		return sigmoidDerivative
	case ActivationFuncNameTanh:
		return tanhDerivative
	default:
		return relu
	}
}

func (h HyperParameters) generateNodes(featureCount uint) []uint {
	nodes := make([]uint, len(h.layers)+1)
	nodes[0] = featureCount
	for i, layer := range h.layers {
		nodes[i+1] = layer.neurons
	}
	return nodes
}

func relu(z float64) float64 {
	if z > 0 {
		return z
	}
	return 0
}

func reluDerivative(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoidDerivative(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

func tanh(z float64) float64 {
	return math.Tanh(z)
}

func tanhDerivative(z float64) float64 {
	return 1 - math.Pow(tanh(z), 2)
}

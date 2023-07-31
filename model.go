package main

import (
	"fmt"
	"math"

	"github.com/codehex/neuralnet/pkg/mx"
)

type TrainedModel struct {
	params *parameters
}

type parameters struct {
	W, b []mx.Matrix
}

type cacheEntry struct {
	Z, A, DW, Db, DZ, DA mx.Matrix
}

func (h HyperParameters) TrainModel(trainingDataSet *ImageSet) (*TrainedModel, error) {
	nodes := h.generateNodes(trainingDataSet.featureCount)
	params := h.initParameters(nodes)

	// Initialize the caches by generating all Z and A matrices, with A[0] being the training data set
	// Empty entries will be added for entites that don't make sense i.e. W[0]
	cache := make([]cacheEntry, len(nodes))
	cache[0].A = trainingDataSet.X()
	m := trainingDataSet.NumberOfExamples()
	L := len(h.layers)
	Y := trainingDataSet.Y()
	OneMinusY := mx.NewZeroMatrix(nodes[L], m)
	OneMinusY.ElemOp(Y, func(v float64) float64 { return float64(1) - v })
	OneMinusAL := mx.NewZeroMatrix(nodes[L], m)
	P1 := mx.NewZeroMatrix(nodes[L], m)
	P2 := mx.NewZeroMatrix(nodes[L], m)
	for i := 1; i <= L; i++ {
		cache[i].Z = mx.NewZeroMatrix(nodes[i], m)
		cache[i].A = mx.NewZeroMatrix(nodes[i], m)
		cache[i].DW = mx.NewZeroMatrix(nodes[i], nodes[i-1])
		cache[i].Db = mx.NewZeroMatrix(nodes[i], 1)
		cache[i].DZ = mx.NewZeroMatrix(nodes[i], m)
		cache[i].DA = mx.NewZeroMatrix(nodes[i], m)
	}

	for iter := uint(0); iter < h.iterations; iter++ {

		// Forward propagation
		for i := 1; i <= L; i++ {
			h.forwardPropagation(cache, params, i)
		}

		// Set up the cache for the last layer
		// dAL = - P1 + P2 where P1 = Y / AL and P2 = (1 - Y) / (1 - AL)
		OneMinusAL.ElemOp(cache[L].A, func(v float64) float64 { return float64(1) - v })
		P2.MatrixElemOp(OneMinusY, OneMinusAL, func(v1, v2 float64) float64 { return v1 / v2 })
		P1.MatrixElemOp(Y, cache[L].A, func(v1, v2 float64) float64 { return v1 / v2 })
		cache[L].DA.MatrixElemOp(P2, P1, func(v1, v2 float64) float64 { return v1 - v2 })

		// Print the cost every 100 iterations
		if iter != 0 && iter%100 == 0 {
			fmt.Println("iter:", iter, ", cost", costFunction(cache[L].A, Y))
		}

		// Backward propagation and update parameters
		for i := L; i > 0; i-- {
			h.backwardPropagation(cache, params, i)
			h.updateParameters(cache, params, i, h.layers[i])
		}
	}
	return &TrainedModel{params: params}, nil
}

func (h HyperParameters) initParameters(nodes []uint) *parameters {
	// We store the weights and biases as indexed by layer number, so we store an empty matrics for
	// layer 0 (the input layer)
	params := parameters{
		W: make([]mx.Matrix, len(nodes)),
		b: make([]mx.Matrix, len(nodes)),
	}
	for i := 1; i < len(nodes); i++ {
		params.W[i] = mx.NewRandomMatrix(nodes[i], nodes[i-1], h.initFactor)
		params.b[i] = mx.NewZeroMatrix(nodes[i], 1)
	}
	return &params
}

func (h HyperParameters) forwardPropagation(cache []cacheEntry, params *parameters, i int) {
	cache[i].Z.MatrixMultiply(params.W[i], cache[i-1].A)
	cache[i].Z.AddColumnVector(cache[i].Z, params.b[i])
	cache[i].A.ElemOp(cache[i].Z, h.Layer(i).ActicationFunc())
}

func costFunction(A, Y mx.Matrix) float64 {
	_, m := Y.Dims()
	sum := float64(0)
	for i := 0; i < m; i++ {
		sum += (Y.At(0, i) * math.Log(A.At(0, i))) + ((1 - Y.At(0, i)) * math.Log(1-A.At(0, i)))
	}
	return -sum / float64(m)
}

func (h HyperParameters) backwardPropagation(cache []cacheEntry, params *parameters, i int) {
	cache[i].DZ.ElemOp(cache[i].Z, h.Layer(i).ActivationDerivativeFunc())
	cache[i].DZ.MatrixElemOp(cache[i].DZ, cache[i].DA, func(v1, v2 float64) float64 { return v1 * v2 })
	cache[i].DW.MatrixMultiply(cache[i].DZ, cache[i-1].A.T())
	cache[i].Db.RowSum(cache[i].DZ, true)
	if i > 1 {
		cache[i-1].DA.MatrixMultiply(params.W[i].T(), cache[i].DZ)
	}
}

func (h HyperParameters) updateParameters(cache []cacheEntry, params *parameters, i int, layer LayerDefinition) {
	deltaFunc := func(x, dx float64) float64 { return x - h.learningRate*dx }
	params.W[i].MatrixElemOp(params.W[i], cache[i].DW, deltaFunc)
	params.b[i].MatrixElemOp(params.b[i], cache[i].Db, deltaFunc)
}

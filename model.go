package neuralnet

import (
	"fmt"
	"math"

	"github.com/codehex/neuralnet/mx"
)

type TrainedModel struct {
	hyper  HyperParameters
	params *parameters
}

type parameters struct {
	W, b []mx.Matrix
}

type velocity struct {
	vW, vb []mx.Matrix
}

type cacheLayer struct {
	Z, A, D, DW, Db, DZ, DA mx.Matrix
}

type batch struct {
	X, Y  mx.MatrixViewable
	cache []cacheLayer
}

func (h HyperParameters) partitionSamples(batchSize uint, nodes []uint, set *ImageSet) []batch {
	if batchSize == 0 || batchSize > set.NumberOfExamples() {
		return []batch{
			{X: set.X(), Y: set.Y(), cache: h.initCache(nodes, set.NumberOfExamples())},
		}
	}

	numOfBatches := set.NumberOfExamples() / batchSize
	if set.NumberOfExamples()%batchSize != 0 {
		numOfBatches++
	}
	batches := make([]batch, numOfBatches)
	for i := uint(0); i < numOfBatches; i++ {
		start := i * batchSize
		end := (i + 1) * batchSize
		if end > set.NumberOfExamples() {
			end = set.NumberOfExamples()
		}
		batches[i] = batch{
			X:     set.X().SliceColumns(int(start), int(end)),
			Y:     set.Y().SliceColumns(int(start), int(end)),
			cache: h.initCache(nodes, end-start),
		}
	}
	return batches
}

func (h HyperParameters) TrainModel(trainingDataSet *ImageSet) (*TrainedModel, error) {
	nodes := h.generateNodes(trainingDataSet.featureCount)
	params := h.initParameters(nodes)
	m := trainingDataSet.NumberOfExamples()
	batches := h.partitionSamples(h.miniBatchSize, nodes, trainingDataSet)
	L := len(nodes) - 1
	v := h.initVelocity(nodes)

	for iter := uint(0); iter < h.iterations; iter++ {
		for batchIndex, batch := range batches {
			// Forward propagation
			for i := 1; i < len(nodes); i++ {
				h.forwardPropagation(batch.X, batch.cache, params, i, true)
			}

			// Set up the cache for the last layer
			// dAL = - P1 + P2 where P1 = Y / AL and P2 = (1 - Y) / (1 - AL)
			batch.cache[L].DA.MatrixElemOp(batch.Y, batch.cache[L].A, func(y, a float64) float64 {
				return (-y / a) + ((1 - y) / (1 - a))
			})

			// Print the cost every 100 iterations
			if iter != 0 && iter%100 == 0 {
				fmt.Println("iter:", iter, ", batch:", batchIndex, ", cost", h.costFunction(batch.cache[L].A, batch.Y, params.W[L]))
			}

			// Backward propagation and update parameters
			for i := L; i > 0; i-- {
				h.backwardPropagation(batch.X, batch.cache, params, i, m)
				h.updateParameters(batch.cache, params, v, i)
			}
		}
	}
	return &TrainedModel{hyper: h, params: params}, nil
}

func (h HyperParameters) initParameters(nodes []uint) *parameters {
	// We store the weights and biases as indexed by layer number, so we store an empty matrics for
	// layer 0 (the input layer)
	params := parameters{
		W: make([]mx.Matrix, len(nodes)),
		b: make([]mx.Matrix, len(nodes)),
	}
	for i := 1; i < len(nodes); i++ {
		params.W[i] = mx.NewRandomMatrix(nodes[i], nodes[i-1], h.Layer(i).initFactor(nodes[i-1]))
		params.b[i] = mx.NewZeroMatrix(nodes[i], 1)
	}
	return &params
}

func (h HyperParameters) initVelocity(nodes []uint) *velocity {
	vel := velocity{
		vW: make([]mx.Matrix, len(nodes)),
		vb: make([]mx.Matrix, len(nodes)),
	}
	for i := 1; i < len(nodes); i++ {
		vel.vW[i] = mx.NewZeroMatrix(nodes[i], nodes[i-1])
		vel.vb[i] = mx.NewZeroMatrix(nodes[i], 1)
	}
	return &vel
}

func (h HyperParameters) initCache(nodes []uint, numOfExamples uint) []cacheLayer {
	cache := make([]cacheLayer, len(nodes))
	// Initialize the caches by generating all Z, A and delta matrices, with A[0] being the training data set
	// Empty entries will be added for entities that don't make sense i.e. W[0]
	m := numOfExamples
	for i := 1; i < len(cache); i++ {
		cache[i].Z = mx.NewZeroMatrix(nodes[i], m)
		cache[i].A = mx.NewZeroMatrix(nodes[i], m)
		cache[i].DW = mx.NewZeroMatrix(nodes[i], nodes[i-1])
		cache[i].Db = mx.NewZeroMatrix(nodes[i], 1)
		cache[i].DZ = mx.NewZeroMatrix(nodes[i], m)
		cache[i].DA = mx.NewZeroMatrix(nodes[i], m)
	}
	return cache
}

func (h HyperParameters) forwardPropagation(X mx.MatrixViewable, cache []cacheLayer, params *parameters, i int, training bool) {
	if i == 1 {
		cache[i].Z.MatrixMultiply(params.W[i], X)
	} else {
		cache[i].Z.MatrixMultiply(params.W[i], cache[i-1].A)
	}
	cache[i].Z.AddColumnVector(cache[i].Z, params.b[i])
	cache[i].A.ElemOp(cache[i].Z, h.Layer(i).ActivationFunc())
	// Only knock out neurons if we're not on the last layer
	if h.keepProb != 0 && i != len(cache)-1 && training {
		// Random generate a matrix with the same dimensions as A[i], set to either 0 or 1
		// with probability keepProb
		rows, cols := cache[i].A.Dims()
		cache[i].D = mx.NewRandomUnitMatrix(uint(rows), uint(cols), h.keepProb)
		// Scale up existing values by 1/keepProb
		cache[i].D.ElemOp(cache[i].D, func(v float64) float64 { return v / h.keepProb })
		// Knock out some of the neurons, boosting the ones that are there
		cache[i].A.MatrixElemOp(cache[i].A, cache[i].D, func(v1, v2 float64) float64 { return v1 * v2 })
	}
}

func (h HyperParameters) costFunction(A, Y, W mx.MatrixViewable) float64 {
	_, m := Y.Dims()
	sum := float64(0)
	for i := 0; i < m; i++ {
		sum += (Y.At(0, i) * math.Log(A.At(0, i))) + ((1 - Y.At(0, i)) * math.Log(1-A.At(0, i)))
	}
	if h.regularizationFactor != 0 {
		sum += (h.regularizationFactor / 2) * W.FrobeniusNorm()
	}
	return -sum / float64(m)
}

func (h HyperParameters) backwardPropagation(X mx.MatrixViewable, cache []cacheLayer, params *parameters, i int, m uint) {
	cache[i].DZ.ElemOp(cache[i].Z, h.Layer(i).ActivationDerivativeFunc())
	cache[i].DZ.MatrixElemOp(cache[i].DZ, cache[i].DA, func(v1, v2 float64) float64 { return v1 * v2 })
	if i == 1 {
		cache[i].DW.MatrixMultiply(cache[i].DZ, X.Transpose())
	} else {
		cache[i].DW.MatrixMultiply(cache[i].DZ, cache[i-1].A.Transpose())
	}
	cache[i].Db.RowSum(cache[i].DZ, true)
	if i > 1 {
		cache[i-1].DA.MatrixMultiply(params.W[i].Transpose(), cache[i].DZ)
	}
	if h.regularizationFactor != 0 {
		cache[i].DW.MatrixElemOp(cache[i].DW, params.W[i], func(v1, v2 float64) float64 {
			return v1 + (h.regularizationFactor * v2)
		})
	}
	if h.keepProb != 0 && i != len(cache)-1 && i != 1 {
		cache[i-1].DA.MatrixElemOp(cache[i-1].DA, cache[i].D, func(v1, v2 float64) float64 { return v1 * v2 })
	}
	cache[i].DW.ElemOp(cache[i].DW, func(v float64) float64 { return v / float64(m) })
}

func (h HyperParameters) updateParameters(cache []cacheLayer, params *parameters, v *velocity, i int) {
	deltaFunc := func(x, dx float64) float64 { return x - h.learningRate*dx }
	if h.momentumBeta == 0 {
		params.W[i].MatrixElemOp(params.W[i], cache[i].DW, deltaFunc)
		params.b[i].MatrixElemOp(params.b[i], cache[i].Db, deltaFunc)
		return
	}

	// Update the momentum and use that to update the parameters
	updateMomentFunc := func(v1, v2 float64) float64 {
		return (h.momentumBeta * v1) + ((1 - h.momentumBeta) * v2)
	}

	v.vW[i].MatrixElemOp(v.vW[i], cache[i].DW, updateMomentFunc)
	v.vb[i].MatrixElemOp(v.vb[i], cache[i].Db, updateMomentFunc)
	params.W[i].MatrixElemOp(params.W[i], v.vW[i], deltaFunc)
	params.b[i].MatrixElemOp(params.b[i], v.vb[i], deltaFunc)
}

func (t *TrainedModel) Predict(set *ImageSet) {
	nodes := t.hyper.generateNodes(set.featureCount)
	params := t.params
	cache := t.hyper.initCache(nodes, set.NumberOfExamples())
	L := len(nodes) - 1
	Y := set.Y()
	m := set.NumberOfExamples()

	for i := 1; i < len(nodes); i++ {
		t.hyper.forwardPropagation(set.X(), cache, params, i, false)
	}

	var correct uint
	var incorrect uint
	for i := 0; i < int(m); i++ {
		var predicted float64
		if cache[L].A.At(0, i) > 0.5 {
			predicted = 1
		} else {
			predicted = 0
		}
		if predicted == Y.At(0, i) {
			correct++
		} else {
			incorrect++
		}
	}

	fmt.Println("correct:", correct, ", incorrect:", incorrect, ", accuracy:", float64(correct)/float64(m))

}

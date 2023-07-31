package mx

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
	transposed bool
	imp        *mat.Dense
}

func NewRandomMatrix(rows, columns uint, randomFactor float64) Matrix {
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	result := make([]float64, rows*columns)
	for i := range result {
		result[i] = r.Float64() * randomFactor
	}
	return Matrix{imp: mat.NewDense(int(rows), int(columns), result)}
}

func NewZeroMatrix(rows, columns uint) Matrix {
	return Matrix{imp: mat.NewDense(int(rows), int(columns), nil)}
}

func NewHorizontalStackedMatrix(vectors [][]float64) Matrix {
	result := mat.NewDense(len(vectors[0]), len(vectors), nil)
	for j := 0; j < len(vectors); j++ {
		for i := 0; i < len(vectors[j]); i++ {
			result.Set(i, j, vectors[j][i])
		}
	}
	return Matrix{imp: result}
}

func NewColumnVector(values []float64) Matrix {
	return Matrix{imp: mat.NewDense(len(values), 1, values)}
}

func NewRowVector(values []float64) Matrix {
	return Matrix{imp: mat.NewDense(1, len(values), values)}
}

func (m Matrix) Dims() (rows, columns int) {
	r, c := m.imp.Dims()
	if m.transposed {
		return c, r
	}
	return r, c
}

func (m Matrix) At(row, column int) float64 {
	if m.transposed {
		return m.imp.At(column, row)
	}
	return m.imp.At(row, column)
}

func (m Matrix) MatrixMultiply(a, b Matrix) {
	switch {
	case a.transposed && b.transposed:
		m.imp.Mul(a.imp.T(), b.imp.T())
	case a.transposed:
		m.imp.Mul(a.imp.T(), b.imp)
	case b.transposed:
		m.imp.Mul(a.imp, b.imp.T())
	default:
		m.imp.Mul(a.imp, b.imp)
	}
}

func (m Matrix) AddColumnVector(a, b Matrix) {
	if a.transposed || b.transposed {
		panic("Cannot add column vector to transposed matrix")
	}
	m.imp.Apply(func(i, j int, v float64) float64 {
		return v + b.imp.At(i, 0)
	}, a.imp)
}

func (m Matrix) ElemOp(a Matrix, f func(v float64) float64) {
	applyFunc := func(i, j int, v float64) float64 {
		return f(v)
	}
	m.imp.Apply(applyFunc, a.imp)
}

func (m Matrix) MatrixElemOp(a, b Matrix, f func(v1, v2 float64) float64) {
	if a.transposed || b.transposed {
		panic("Cannot element-wise operate on transposed matrix")
	}
	applyFunc := func(i, j int, v float64) float64 {
		return f(v, b.imp.At(i, j))
	}
	m.imp.Apply(applyFunc, a.imp)
}

func (m Matrix) T() Matrix {
	return Matrix{imp: m.imp, transposed: !m.transposed}
}

func (m Matrix) RowSum(matToSum Matrix, normalize bool) {
	rRes, cRes := m.Dims()
	r, c := matToSum.Dims()
	if cRes != 1 || rRes != r {
		panic("Result matrix must be a column vector of the same length as the number of rows in the matrix")
	}
	var wg sync.WaitGroup

	wg.Add(r)

	for i := 0; i < r; i++ {
		go func(i int) {
			defer wg.Done()
			sum := 0.0
			for j := 0; j < c; j++ {
				sum += matToSum.imp.At(i, j)
			}
			if normalize {
				sum /= float64(c)
			}
			m.imp.Set(i, 0, sum)
		}(i)
	}
	wg.Wait()
}

func (m Matrix) String() string {
	return fmt.Sprintf("%+v", *m.imp)
}

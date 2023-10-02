package mx

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
	*mat.Dense
}

type MatrixView struct {
	mat.Matrix
}

func (m Matrix) View() MatrixView {
	return MatrixView{m.Dense}
}

type MatrixViewable interface {
	At(row, column int) float64
	Dims() (rows, columns int)
	Transpose() MatrixView
	View() MatrixView
}

func (m Matrix) Dims() (rows, columns int) {
	return m.Dense.Dims()
}

func (m Matrix) At(row, column int) float64 {
	return m.Dense.At(row, column)
}

func (m Matrix) Transpose() MatrixView {
	return MatrixView{m.Dense.T()}
}

func (v MatrixView) Dims() (rows, columns int) {
	return v.Matrix.Dims()
}

func (v MatrixView) At(row, column int) float64 {
	return v.Matrix.At(row, column)
}

func (v MatrixView) Transpose() MatrixView {
	return MatrixView{v.Matrix.T()}
}

func (v MatrixView) View() MatrixView {
	return v
}

func NewRandomMatrix(rows, columns uint, randomFactor float64) Matrix {
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	result := make([]float64, rows*columns)
	for i := range result {
		result[i] = r.Float64() * randomFactor
	}
	return Matrix{mat.NewDense(int(rows), int(columns), result)}
}

func NewRandomUnitMatrix(rows, columns uint, prob float64) Matrix {
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	result := make([]float64, rows*columns)
	for i := range result {
		if r.Float64() < prob {
			result[i] = 1
		}
	}
	return Matrix{mat.NewDense(int(rows), int(columns), result)}
}

func NewZeroMatrix(rows, columns uint) Matrix {
	return Matrix{mat.NewDense(int(rows), int(columns), nil)}
}

func NewHorizontalStackedMatrix(vectors [][]float64) Matrix {
	result := mat.NewDense(len(vectors[0]), len(vectors), nil)
	for j := 0; j < len(vectors); j++ {
		for i := 0; i < len(vectors[j]); i++ {
			result.Set(i, j, vectors[j][i])
		}
	}
	return Matrix{result}
}

func NewColumnVector(values []float64) Matrix {
	return Matrix{mat.NewDense(len(values), 1, values)}
}

func NewRowVector(values []float64) Matrix {
	return Matrix{mat.NewDense(1, len(values), values)}
}

func (m Matrix) MatrixMultiply(a, b MatrixViewable) {
	m.Mul(a.View().Matrix, b.View().Matrix)
}

func (m Matrix) AddColumnVector(a, b MatrixViewable) {
	m.Apply(func(i, j int, v float64) float64 {
		return v + b.At(i, 0)
	}, a.View().Matrix)
}

func (m Matrix) ElemOp(a MatrixViewable, f func(v float64) float64) {
	applyFunc := func(i, j int, v float64) float64 {
		return f(v)
	}
	m.Apply(applyFunc, a.View().Matrix)
}

func (m Matrix) MatrixElemOp(a, b MatrixViewable, f func(v1, v2 float64) float64) {
	applyFunc := func(i, j int, v float64) float64 {
		return f(v, b.At(i, j))
	}
	m.Apply(applyFunc, a.View().Matrix)
}

func (m Matrix) RowSum(matToSum MatrixViewable, normalize bool) {
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
				sum += matToSum.At(i, j)
			}
			if normalize {
				sum /= float64(c)
			}
			m.Set(i, 0, sum)
		}(i)
	}
	wg.Wait()
}

func (m Matrix) FrobeniusNorm() float64 {
	r, c := m.Dims()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += m.At(i, j) * m.At(i, j)
		}
	}
	return sum
}

func (m Matrix) String() string {
	return fmt.Sprintf("%+v", *m.Dense)
}

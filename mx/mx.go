package mx

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
	imp *mat.Dense
}

type MatrixView struct {
	view mat.Matrix
}

func (m Matrix) View() MatrixView {
	return MatrixView{view: m.imp}
}

type MatrixViewable interface {
	At(row, column int) float64
	Dims() (rows, columns int)
	Transpose() MatrixView
	View() MatrixView
	FrobeniusNorm() float64
}

func (m Matrix) Dims() (rows, columns int) {
	return m.imp.Dims()
}

func (m Matrix) At(row, column int) float64 {
	return m.imp.At(row, column)
}

func (m Matrix) Transpose() MatrixView {
	return MatrixView{m.imp.T()}
}

func (v MatrixView) Dims() (rows, columns int) {
	return v.view.Dims()
}

func (v MatrixView) At(row, column int) float64 {
	return v.view.At(row, column)
}

func (v MatrixView) Transpose() MatrixView {
	return MatrixView{v.view.T()}
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
	m.imp.Mul(a.View().view, b.View().view)
}

func (m Matrix) AddColumnVector(a, b MatrixViewable) {
	m.imp.Apply(func(i, j int, v float64) float64 {
		return v + b.At(i, 0)
	}, a.View().view)
}

func (m Matrix) ElemOp(a MatrixViewable, f func(v float64) float64) {
	applyFunc := func(i, j int, v float64) float64 {
		return f(v)
	}
	m.imp.Apply(applyFunc, a.View().view)
}

func (m Matrix) MatrixElemOp(a, b MatrixViewable, f func(v1, v2 float64) float64) {
	applyFunc := func(i, j int, v float64) float64 {
		return f(v, b.At(i, j))
	}
	m.imp.Apply(applyFunc, a.View().view)
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
			m.imp.Set(i, 0, sum)
		}(i)
	}
	wg.Wait()
}

func (m Matrix) FrobeniusNorm() float64 {
	return m.View().FrobeniusNorm()
}

func (v MatrixView) FrobeniusNorm() float64 {
	r, c := v.Dims()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += v.At(i, j) * v.At(i, j)
		}
	}
	return sum
}

func (m Matrix) SliceColumns(start, end int) MatrixView {
	_, c := m.Dims()
	if start < 0 || end > c {
		panic("Slice out of bounds")
	}
	return MatrixView{view: m.imp.Slice(0, m.imp.RawMatrix().Rows, start, end)}
}

func (m Matrix) String() string {
	return fmt.Sprintf("%+v", *m.imp)
}

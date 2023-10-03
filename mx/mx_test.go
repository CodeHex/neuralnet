package mx_test

import (
	"testing"

	"github.com/codehex/neuralnet/mx"
)

func TestNewRandomMatrix(t *testing.T) {
	m := mx.NewRandomMatrix(3, 4, 0.01)
	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Errorf("Expected matrix dimensions to be (3, 4), but got (%d, %d)", r, c)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if m.At(i, j) > 0.01 {
				t.Errorf("Expected matrix value at (%v, %v) to be less than 0.01, but got %v", i, j, m.At(i, j))
			}
			if m.At(i, j) < 0 {
				t.Errorf("Expected matrix value at (%v, %v) to be greater than 0, but got %v", i, j, m.At(i, j))
			}
		}
	}
}

func TestNewZeroMatrix(t *testing.T) {
	m := mx.NewZeroMatrix(3, 4)
	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Errorf("Expected matrix dimensions to be (3, 4), but got (%d, %d)", r, c)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if m.At(i, j) != 0 {
				t.Errorf("Expected matrix value at (%v, %v) to be 0, but got %v", i, j, m.At(i, j))
			}
		}
	}
}

func TestNewRandomUnitMatrix(t *testing.T) {
	m := mx.NewRandomUnitMatrix(3, 4, 0.5)
	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Errorf("Expected matrix dimensions to be (3, 4), but got (%d, %d)", r, c)
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if m.At(i, j) != 0 && m.At(i, j) != 1 {
				t.Errorf("Expected matrix value at (%v, %v) to be 0 or 1, but got %v", i, j, m.At(i, j))
			}
		}
	}
}

func TestNewHorizontalStackedMatrix(t *testing.T) {
	data := [][]float64{
		{1.1, 2.2, 3.3},
		{4.4, 5.5, 6.6},
	}

	m := mx.NewHorizontalStackedMatrix(data)
	r, c := m.Dims()
	if r != 3 || c != 2 {
		t.Errorf("Expected matrix dimensions to be (3, 2), but got (%d, %d)", r, c)
	}
	if m.At(0, 0) != 1.1 {
		t.Errorf("Expected matrix value at (0, 0) to be 1.1, but got %v", m.At(0, 0))
	}
	if m.At(1, 0) != 2.2 {
		t.Errorf("Expected matrix value at (1, 0) to be 2.2, but got %v", m.At(1, 0))
	}
	if m.At(2, 0) != 3.3 {
		t.Errorf("Expected matrix value at (2, 0) to be 3.3, but got %v", m.At(2, 0))
	}
	if m.At(0, 1) != 4.4 {
		t.Errorf("Expected matrix value at (0, 1) to be 4.4, but got %v", m.At(0, 1))
	}
	if m.At(1, 1) != 5.5 {
		t.Errorf("Expected matrix value at (1, 1) to be 5.5, but got %v", m.At(1, 1))
	}
	if m.At(2, 1) != 6.6 {
		t.Errorf("Expected matrix value at (2, 1) to be 6.6, but got %v", m.At(2, 1))
	}
}

func TestRowSum(t *testing.T) {
	m := mx.NewZeroMatrix(2, 3)
	m.Set(0, 0, 1)
	m.Set(0, 1, 2)
	m.Set(0, 2, 3)
	m.Set(1, 0, 4)
	m.Set(1, 1, 5)
	m.Set(1, 2, 6)
	target := mx.NewZeroMatrix(2, 1)
	target.RowSum(m, false)

	if target.At(0, 0) != 6 {
		t.Errorf("Expected matrix value at (0, 0) to be 6, but got %v", m.At(0, 0))
	}
	if target.At(1, 0) != 15 {
		t.Errorf("Expected matrix value at (1, 0) to be 15, but got %v", m.At(1, 0))
	}
}

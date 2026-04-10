/*
1. Реализовать LU-разложение матрицы A c выбором ведущего элемента по по столбцу
или по всей матрице. Проверить разложение сравнением матриц LU и P A (или P AQ),
где P — матрица перестановки строк, а Q — столбцов. Выполнить для системы
произвольной размерности, генерировать случайную матрицу для демонстрации работы программы.
С использованием LU-разложения найти:
a) Определитель матрицы A;
b) Решение СЛАУ Ax = b, выполнить проверку равенства Ax − b = 0;
c) Матрицу A−1
(выполнить проверку AA−1 и A−1A);
d) Число обусловленности матрицы A.
*/

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Matrix [][]float64

// создадим пустую матрицу
func NewMatrix(n int) Matrix {
	m := make(Matrix, n)
	for i := range m {
		m[i] = make([]float64, n)

	}
	return m
}

// копирование матрицы
func (m Matrix) Copy() Matrix {
	n := len(m)
	copyM := NewMatrix(n)
	for i := 0; i < n; i++ {
		copy(copyM[i], m[i])
	}
	return copyM
}

// Умножение матриц
func Multiply(A Matrix, B Matrix) Matrix {
	n := len(A)
	C := NewMatrix(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

// Умножение матрицы на вектор
func MultiplyVec(A Matrix, b []float64) []float64 {
	n := len(A)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			y[i] += A[i][j] * b[j]
		}
	}
	return y
}

// LU-разложение с выбором ведущего элемента по столбцу (PA=LU)
func LUDecomposition(A Matrix) (L, U, P Matrix, swaps int) {
	n := len(A)
	L = NewMatrix(n)
	U = A.Copy()
	P = NewMatrix(n)
	pVector := make([]int, n)

	for i := 0; i < n; i++ {
		pVector[i] = i
		L[i][i] = 1.0
	}

	for i := 0; i < n; i++ {
		//выберем ведущий элемент
		maxVal := math.Abs(U[i][i])
		pivotRow := i

		for k := i + 1; k < n; k++ {
			if math.Abs(U[k][i]) > maxVal {
				maxVal = math.Abs(U[k][i])
				pivotRow = k
			}
		}

		if pivotRow != i {
			U[i], U[pivotRow] = U[pivotRow], U[i]
			pVector[i], pVector[pivotRow] = pVector[pivotRow], pVector[i]

			//переставновка в L
			for k := 0; k < n; k++ {
				L[i][k], L[pivotRow][k] = L[pivotRow][k], L[i][k]
			}
			swaps++
		}

		for j := i + 1; j < n; j++ {
			L[j][i] = U[j][i] / U[i][i]
			for k := i; k < n; k++ {
				U[j][k] -= L[j][i] * U[i][k]
			}
		}

	}

	for i, rowIdx := range pVector {
		P[i][rowIdx] = 1
	}
	return
}

// Решение Ly=b
func forwardSubstitution(L Matrix, b []float64) []float64 {
	n := len(L)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < i; j++ {
			sum += L[i][j] * y[j]
		}
		y[i] = b[i] - sum
	}
	return y
}

// Решение Ux=y
func backwardSubstitution(U Matrix, y []float64) []float64 {
	n := len(U)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += U[i][j] * x[j]
		}
		x[i] = (y[i] - sum) / U[i][i]
	}
	return x
}

// Норма матрицы
func (m Matrix) Norm() float64 {
	max := 0.0
	for _, row := range m {
		sum := 0.0
		for _, val := range row {
			sum += math.Abs(val)
		}
		if sum > max {
			max = sum
		}
	}
	return max
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var n int
	fmt.Print("Введите размер матрицы: ")
	fmt.Scan(&n)

	A := NewMatrix(n)
	b := make([]float64, n)

	fmt.Println("Матрица А: ")
	for i := 0; i < n; i++ {
		b[i] = float64(rand.Int63n(100))
		for j := 0; j < n; j++ {
			A[i][j] = float64(rand.Int63n(100))
			fmt.Printf("%4.0f ", A[i][j])
		}
		fmt.Println()
	}
	L, U, P, swaps := LUDecomposition(A)

	//определитель
	det := 1.0
	if swaps%2 != 0 {
		det = -1.0
	}
	for i := 0; i < n; i++ {
		det *= U[i][i]
	}
	fmt.Printf("Определитель det(A) = %.0f\n", math.Round(det))

	//Проверка разложения PA=LU
	PA := Multiply(P, A)
	LU := Multiply(L, U)

	fmt.Println("Проверка разложения PA=LU: ")
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%10.2e", PA[i][j]-LU[i][j])
		}
		fmt.Println()
	}

	//Решение СЛАУ Ax=b
	Pb := MultiplyVec(P, b)
	y := forwardSubstitution(L, Pb)
	x := backwardSubstitution(U, y)
	fmt.Printf("Решение СЛАУ x = %v\n", x)

	//Проверка Ax-b=0
	fmt.Println("Проверка Ax-b=0: ")
	Ax := MultiplyVec(A, x)
	for i := 0; i < n; i++ {
		fmt.Printf("[%d]: %e\n", i, Ax[i]-b[i])
	}

	//Обратная матрица
	Inv := NewMatrix(n)
	for i := 0; i < n; i++ {
		e := make([]float64, n)
		e[i] = 1.0
		Pe := MultiplyVec(P, e)
		yi := forwardSubstitution(L, Pe)
		xi := backwardSubstitution(U, yi)
		for r := 0; r < n; r++ {
			Inv[r][i] = xi[r]
		}
	}
	fmt.Println("Обратная матрица A^-1: ")
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%8.2f ", Inv[i][j])
		}
		fmt.Println()
	}

	//Проверка обратной матрицы
	fmt.Println("Проверка обратной матрицы A*A^-1 = I: ")
	AAinv := Multiply(A, Inv)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%10.2f ", AAinv[i][j])
		}
		fmt.Println()
	}

	//Число обусловленности
	cond := A.Norm() * Inv.Norm()
	fmt.Printf("Число обусловленности: %.4f\n", cond)

}

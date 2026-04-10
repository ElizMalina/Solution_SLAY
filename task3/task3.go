/*
3. Реализовать QR-разложение матрицы A методом отражений или вращений.
Проверить разложение перемножением матриц Q и R. С его помощью найти решение
невырожденной СЛАУ Ax = b.
*/

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Matrix [][]float64

func NewMatrix(r, c int) Matrix {
	m := make(Matrix, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}

// Умножение матриц
func Multiply(A Matrix, B Matrix) Matrix {
	n := len(A)
	C := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

// Норма вектора
func norm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// QR-разложение методом отражений Хаусхолдера
func QRDecomposition(A Matrix) (Q, R Matrix) {
	n := len(A)
	m := len(A[0])
	R = NewMatrix(n, m)

	for i := 0; i < n; i++ {
		copy(R[i], A[i])
	}

	Q = NewMatrix(n, n)
	for i := 0; i < n; i++ {
		Q[i][i] = 1.0
	}

	for k := 0; k < m && k < n-1; k++ {
		x := make([]float64, n-k)
		for i := k; i < n; i++ {
			x[i-k] = R[i][k]
		}

		xNorm := norm(x)
		if xNorm == 0 {
			continue
		}

		// Выбираем знак для стабильности
		s := -1.0
		if x[0] < 0 {
			s = 1.0
		}

		// Вектор отражения v
		v := make([]float64, n-k)
		copy(v, x)
		v[0] -= s * xNorm

		vNorm := norm(v)
		if vNorm > 1e-15 {
			for i := 0; i < len(v); i++ {
				v[i] /= vNorm
			}
		} else {
			continue
		}

		//R = H * R
		for j := k; j < m; j++ {
			dot := 0.0
			for i := k; i < n; i++ {
				dot += v[i-k] * R[i][j]
			}
			for i := k; i < n; i++ {
				R[i][j] -= 2.0 * v[i-k] * dot
			}
		}

		// Q = Q * H
		for j := 0; j < n; j++ {
			dot := 0.0
			for i := k; i < n; i++ {
				dot += v[i-k] * Q[j][i]
			}
			for i := k; i < n; i++ {
				Q[j][i] -= 2.0 * v[i-k] * dot
			}
		}
	}
	return Q, R
}

// Rx = Q^T * b
func solveQR(Q, R Matrix, b []float64) []float64 {
	n := len(Q)
	//y = Q^T * b
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			y[i] += Q[j][i] * b[j]
		}
	}

	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += R[i][j] * x[j]
		}
		x[i] = (y[i] - sum) / R[i][i]
	}
	return x
}

func main() {
	rand.Seed(time.Now().UnixNano())
	var n int
	fmt.Print("ВВедите размер матрицы n: ")
	fmt.Scan(&n)
	A := NewMatrix(n, n)
	b := make([]float64, n)

	fmt.Println("Матрица A:")
	for i := 0; i < n; i++ {
		b[i] = float64(rand.Int63n(21) - 10)
		for j := 0; j < n; j++ {
			A[i][j] = float64(rand.Int63n(21) - 10)
			fmt.Printf("%8.0f ", A[i][j])
		}
		fmt.Printf(" | %8.0f\n", b[i])
	}

	Q, R := QRDecomposition(A)

	fmt.Println("\nМатрица Q (ортогональная):")
	for _, row := range Q {
		for _, val := range row {
			fmt.Printf("%8.4f ", val)
		}
		fmt.Println()
	}

	fmt.Println("\nМатрица R (верхнетреугольная):")
	for _, row := range R {
		for _, val := range row {
			fmt.Printf("%8.4f ", val)
		}
		fmt.Println()
	}

	//проверка разложения
	CheckA := Multiply(Q, R)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if A[i][j]-CheckA[i][j] < 1e-12 {
				continue
			} else {
				fmt.Printf("Разложение не удалось %4.0f != %4.0f", A[i][j], CheckA[i][j])
			}

		}
	}

	fmt.Println("\nПроверка завершена")

	// Решение СЛАУ
	x := solveQR(Q, R, b)
	fmt.Printf("\nРешение СЛАУ x: %v\n", x)

	// Проверка Ax - b
	fmt.Print("Проверка Ax - b: ")
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += A[i][j] * x[j]
		}
		fmt.Printf("%.2e ", sum-b[i])
	}
	fmt.Println()
}

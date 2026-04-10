/*
2. Модифицировать алгоритм для нахождения ранга вырожденных матриц: при выборе
ведущего элемента только по столбцу надо приводить матрицу к ступенчатой форме,
что потребует обнулять элементы не только под диагональю; при выборе ведущего
элемента по всей матрице никаких изменений не требуется — в этом случае матрица
приводится к трапецевидной форме). Проверять так же систему с вырожденной матрицей
на совместность и выдавать любое частное решение, если она совместна
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
func NewMatrix(rows int, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)

	}
	return m
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

func SolveAndRank(A Matrix, b []float64) (int, []float64, bool) {
	n := len(A)
	//расширенная матрица
	aug := NewMatrix(n, n+1)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i][j] = A[i][j]
		}
		aug[i][n] = b[i]
	}

	rank := 0
	pivotRow := 0
	pivotCols := make([]int, n)
	for i := range pivotCols {
		pivotCols[i] = -1
	}

	//ступенчатая форма
	for j := 0; j < n && pivotRow < n; j++ {
		maxVal := math.Abs(aug[pivotRow][j])
		selRow := pivotRow
		for k := pivotRow + 1; k < n; k++ {
			if math.Abs(aug[k][j]) > maxVal {
				maxVal = math.Abs(aug[k][j])
				selRow = k
			}
		}
		if maxVal < 1e-12 {
			continue //ранг не растет, идем дальше
		}

		aug[pivotRow], aug[selRow] = aug[selRow], aug[pivotRow]

		for k := 0; k < n; k++ {
			if k != pivotRow {
				factor := aug[k][j] / aug[pivotRow][j]
				for l := j; l <= n; l++ {
					aug[k][l] -= factor * aug[pivotRow][l]
				}
			}
		}

		pivotCols[pivotRow] = j
		pivotRow++
		rank++
	}

	// Проверка на совместность Теорема Кронекера-Капелли
	for i := pivotRow; i < n; i++ {
		if math.Abs(aug[i][n]) > 1e-12 {
			return rank, nil, false
		}
	}

	// Частное решение
	x := make([]float64, n)
	for i := 0; i < pivotRow; i++ {
		col := pivotCols[i]
		x[col] = aug[i][n] / aug[i][col]
	}

	return rank, x, true
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var n int
	fmt.Print("Введите размер матрицы n: ")
	fmt.Scan(&n)

	A := NewMatrix(n, n)
	b := make([]float64, n)

	fmt.Println("Матрица А: ")
	for i := 0; i < n-1; i++ {
		if i == n-1 && n > 1 {
			//создаем зависимую строку
			for j := 0; j < n; j++ {
				A[i][j] = A[0][j] + A[1][j]
			}
			b[i] = b[0] + b[1]
			for j := 0; j < n; j++ {
				fmt.Printf("%4.0f", A[i][j])
			}
			fmt.Printf("| %4.0f Линейно зависимая\n", b[i])
		} else {
			b[i] = float64(rand.Intn(11))
			for j := 0; j < n; j++ {
				A[i][j] = float64(rand.Intn(11))
				fmt.Printf("%4.0f ", A[i][j])
			}
			fmt.Printf(" | %4.0f\n", b[i])
		}
	}

	// РЕШЕНИЕ
	rank, x, consistent := SolveAndRank(A, b)

	fmt.Printf("Ранг матрицы A: %d\n", rank)

	if !consistent {
		fmt.Println("Система несовместна.")
	} else {
		fmt.Printf("Система совместна. Частное решение x: %v\n", x)

		// ПРОВЕРКА Ax - b = 0
		fmt.Println("\nПроверка Ax - b = 0:")
		Ax := MultiplyVec(A, x)
		for i := 0; i < n; i++ {
			fmt.Printf("   [%d]: %e\n", i, Ax[i]-b[i])
		}
	}
}

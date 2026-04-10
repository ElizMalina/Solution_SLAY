/*4.
Реализовать метод Якоби и метод Зейделя решения СЛАУ. Сравнить
на примере СЛАУ с матрицей с диагональным преобладанием и с положительно
определённой матрицей без диагонального преобладания (генерировать
случайные матрицы по размерности). Дать априорную оценку числа необходимых итераций,
сравнить с апостериорной оценкой.
*/

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Vector []float64
type Matrix [][]float64

func NewVector(n int) Vector { return make(Vector, n) }

func NewMatrix(n int) Matrix {
	m := make(Matrix, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	return m
}

func (v Vector) Norm() float64 {
	max := 0.0
	for _, x := range v {
		if math.Abs(x) > max {
			max = math.Abs(x)
		}
	}
	return max
}

// Проверка невязки Ax - b
func ComputeResidual(A Matrix, x Vector, b Vector) float64 {
	n := len(A)
	res := NewVector(n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += A[i][j] * x[j]
		}
		res[i] = sum - b[i]
	}
	return res.Norm()
}

func GenDiagDominant(n int) (Matrix, Vector) {
	A := NewMatrix(n)
	b := NewVector(n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			A[i][j] = float64(rand.Intn(10) - 5)
			sum += math.Abs(A[i][j])
		}
		A[i][i] = sum + float64(rand.Intn(5)+5)
		b[i] = float64(rand.Intn(20) - 10)
	}
	return A, b
}

func GenPosDefNoDiag(n int) (Matrix, Vector) {
	M := NewMatrix(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			M[i][j] = float64(rand.Intn(4) + 1)
		}
	}
	A := NewMatrix(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				A[i][j] += M[k][i] * M[k][j]
			}
		}
	}
	b := NewVector(n)
	for i := range b {
		b[i] = float64(rand.Intn(50) - 25)
	}
	return A, b
}

func Jacobi(A Matrix, b Vector, eps float64) (Vector, int) {
	n := len(A)
	x := NewVector(n)
	tempX := NewVector(n)
	it := 0
	for it < 1000 {
		it++
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += A[i][j] * x[j]
				}
			}
			tempX[i] = (b[i] - sum) / A[i][i]
		}
		diff := NewVector(n)
		for i := 0; i < n; i++ {
			diff[i] = tempX[i] - x[i]
		}
		if diff.Norm() < eps {
			copy(x, tempX)
			return x, it
		}
		copy(x, tempX)
	}
	return x, it
}

func Seidel(A Matrix, b Vector, eps float64) (Vector, int) {
	n := len(A)
	x := NewVector(n)
	it := 0
	for it < 1000 {
		it++
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			oldXi := x[i]
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += A[i][j] * x[j]
				}
			}
			x[i] = (b[i] - sum) / A[i][i]
			if d := math.Abs(x[i] - oldXi); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff < eps {
			return x, it
		}
	}
	return x, it
}

func FullAnalysis(A Matrix, b Vector, eps float64) {
	n := len(A)

	fmt.Println("Матрица A | Вектор b:")
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%7.2f ", A[i][j])
		}
		fmt.Printf(" | %7.2f\n", b[i])
	}

	// Расчет нормы q (для априорной оценки)
	q := 0.0
	for i := 0; i < n; i++ {
		rowSum := 0.0
		for j := 0; j < n; j++ {
			if i != j {
				rowSum += math.Abs(A[i][j] / A[i][i])
			}
		}
		if rowSum > q {
			q = rowSum
		}
	}
	fmt.Printf("\nПараметр сходимости q (норма матрицы Якоби): %.4f\n", q)

	// Априорная оценка
	if q < 1 {
		x1 := NewVector(n)
		for i := 0; i < n; i++ {
			x1[i] = b[i] / A[i][i]
		}
		dist := 0.0 // ||x1 - x0||
		for i := 0; i < n; i++ {
			if math.Abs(x1[i]) > dist {
				dist = math.Abs(x1[i])
			}
		}
		//k >= [ln(eps*(1-q)/dist)] / ln(q)
		kPrior := math.Log(eps*(1-q)/dist) / math.Log(q)
		fmt.Printf("Априорная оценка: необходимо минимум %.0f итераций\n", math.Ceil(kPrior))
	} else {
		fmt.Println("Априорная оценка: Невозможна (q >= 1). Метод Якоби может разойтись.")
	}

	solJ, itJ := Jacobi(A, b, eps)
	solS, itS := Seidel(A, b, eps)

	resJ := ComputeResidual(A, solJ, b)
	fmt.Printf("Метод Якоби:   %d итераций | Невязка: %.2e | Решение: %v\n", itJ, resJ, solJ)

	resS := ComputeResidual(A, solS, b)
	fmt.Printf("Метод Зейделя: %d итераций | Невязка: %.2e | Решение: %v\n", itS, resS, solS)

	if itJ >= 1000 {
		fmt.Println("(!) Метод Якоби не сошелся за 1000 итераций.")
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var n int
	fmt.Print("ВВедите размер матрицы n: ")
	fmt.Scan(&n)
	eps := 1e-5

	// Случай 1: Диагональное преобладание (Сходится всё)
	fmt.Println("\nМАТРИЦА С ДИАГОНАЛЬНЫМ ПРЕОБЛАДАНИЕМ")
	A1, b1 := GenDiagDominant(n)
	FullAnalysis(A1, b1, eps)

	// Случай 2: Положительно определенная (Зейдель сойдется, Якоби - под вопросом)
	fmt.Println("\nПОЛОЖИТЕЛЬНО ОПРЕДЕЛЕННАЯ МАТРИЦА")
	A2, b2 := GenPosDefNoDiag(n)
	FullAnalysis(A2, b2, eps)
}

package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

var asciiChars = " .,:;lodx%CO0@#KXWM"

// func main() {
// 	data := make([]float64, 15)
// 	for i := range data {
// 		data[i] = rand.NormFloat64()
// 	}
// 	dense := mat.NewDense(3, 5, data)
// 	printDense(dense)
//
// }

func off() {
	data, m, n := readData()

	matrix := mat.NewDense(m, n, data)
	printDense(matrix)
	println(m, n)

	// data_train := transposeMatrix(matrix)
	data_train := matrix.T()
	printMatrix(data_train)

	Y_train := matrix.ColView(0)
	println(Y_train)
	X_train := slice(matrix, 1, matrix.RawMatrix().Rows)
	printMatrix(X_train)

	fmt.Println(Y_train.Len())
	fmt.Println(X_train.Dims())

	// W1, b1, W2, b2 := initParams()
	// forwardPropagation(W1, b1, W2, b2, X_train)

}

func slice(m *mat.Dense, startIdx int, endIdx int) *mat.Dense {
	cols := m.RawMatrix().Cols
	newRows := endIdx - startIdx

	newMat := mat.NewDense(newRows, cols, nil)

	for i := startIdx; i < endIdx; i++ {
		row := mat.Row(nil, i, m)
		newMat.SetRow(i-startIdx, row)
	}

	return newMat
}

// func main() {
// 	fmt.Println(1.1 * 2)
// }

// func main() {
// 	matrix := readData()
// 	printDigit(matrix)
// }

func readData() ([]float64, int, int) {
	// Open the CSV file
	file, err := os.Open("data/train.csv")
	if err != nil {
		log.Fatal("Error opening file:", err)
	}
	defer file.Close()

	// Create a new CSV reader
	reader := csv.NewReader(file)

	// Skip the header row
	_, err = reader.Read()
	if err != nil {
		log.Fatal("Error reading header:", err)
	}

	// Initialize the matrix
	var matrix []float64

	var rows = 0
	var cols int
	for {
		record, err := reader.Read()
		if err != nil {
			// Check if we've reached the end of the file
			if err.Error() == "EOF" {
				break
			}
			log.Fatal("Error reading row:", err)
		}

		rows++
		cols = len(record)
		for _, value := range record {
			num, err := strconv.Atoi(value)
			if err != nil {
				log.Fatal("Error converting value to int: %s", err)
			}
			matrix = append(matrix, float64(num))
		}

	}

	return matrix, rows, cols
}

func main() {
	W1, b1, W2, b2 := initParams()
	printMatrix(W1)
	printMatrix(b1)
	printMatrix(W2)
	printMatrix(b2)
}

func initParams() (W1 *mat.Dense, b1 mat.Vector, W2 mat.Dense, b2 mat.Vector) {
	W1 = Randn2D(10, 784)
	Dot(a*mat.Dense, b*mat.Dense)
	W1.RawMatrix().Data(W1, 0.5)
	if err != nil {
		log.Fatal(err)
	}
	b1, err = Subtract(Randn2D(10, 1), 0.5)
	if err != nil {
		log.Fatal(err)
	}
	W2, err = Subtract(Randn2D(10, 10), 0.5)
	if err != nil {
		log.Fatal(err)
	}
	b2, err = Subtract(Randn2D(10, 1), 0.5)
	if err != nil {
		log.Fatal(err)
	}

	return W1, b1, W2, b2
}

// func ReLU(Z [][]float64) Matrix {
// 	m, err := Maximum(Z, 0)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	return m.(Matrix)
// }

// func ReLUDeriv(Z Matrix) {
// 	return Z > 0
// }

// func softmax(Z Matrix) Matrix {
// 	m1 := Exp(Z)
// 	s := Sum(m1)
// 	A, err := Divide(m1, s)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	return A
// }

// func forwardPropagation(W1 Matrix, b1 Matrix, W2 Matrix, b2 Matrix, X [][]int) (Matrix, Matrix, Matrix, Matrix) {
// 	Z1, err := DotT(W1, X)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	Z1, err = Add(Z1, b1)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	A1 := ReLU(Z1)
// 	Z2, err := Dot(W2, A1)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	Z2, err = Add(Z2, b2)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	A2 := softmax(A1)
//
// 	return Z1, A1, Z2, A2
// }

func Max(Z []int) int {
	maximum := 0.0
	for _, row := range Z {
		for val := range row {
			math.Max(maximum, float64(val))
		}
	}
	return int(maximum)
}

func ARange(r int) []float64 {
	a := make([]float64, r)
	for i := range r {
		a[i] = float64(i)
	}
	return a

}
func oneHot(Y []int) [][]int {
	oneHotY := Zeros(len(Y), int(Max(Y))+1).([][]int)
	for i, n := range Y {
		oneHotY[i][n] = 1
	}
	// for i, n := range Y {
	// 	t := oneHotY[i].([]float64)
	// 	t[n] = 1
	// }
	return transposeMatrix(oneHotY)
}

// func backPropagation(Z1 Matrix, A1 Matrix, Z2 Matrix, A2 Matrix, W2 Matrix, Y []int, m int) {
// 	oneHotY := oneHot(Y)
// 	dZ2, err := Subtract(A2, oneHotY)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	dZ2dotA1T, err := Dot(dZ2, transposeMatrix(A1))
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	dW2, err := Multiply(dZ2dotA1T, 1/m)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	db2 := 1 / float64(m) * npSum(dZ2)
// 	_, _ = dW2, db2
// }

func SumA(a []float64) float64 {
	sum := 0.0
	for _, n := range a {
		sum += n
	}
	return sum
}

// func main() {
// 	matrix := [][]float64{{1, 2, 3}, {4, 5, 6}}
// 	result := transposeMatrix(matrix)
// 	printMatrix(result)
// }

func transposeMatrix[T any](matrix [][]T) [][]T {
	rows := len(matrix)
	cols := len(matrix[0])

	transposed := make([][]T, cols)
	for i := range transposed {
		transposed[i] = make([]T, rows)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}

	return transposed
}

func printDense(matrix *mat.Dense) {
	fmt.Printf("Matrix dimensions: %d rows x %d columns\n", matrix.RawMatrix().Rows, matrix.RawMatrix().Cols)

	fmt.Println("[")
	for i := 0; i < 5 && i < matrix.RawMatrix().Rows; i++ {
		fmt.Printf("Row %d: [", i)
		for j := 0; j < 785 && j < matrix.RawMatrix().Cols; j++ {
			fmt.Printf("%.2f ", matrix.At(i, j))
		}
		fmt.Println("... ]")
	}
	fmt.Println("]")

}

func printMatrix(m mat.Matrix) {
	r, c := m.Dims()
	fmt.Printf("Matrix dimensions: %d rows x %d columns\n", r, c)

	fmt.Println("[")
	for i := 0; i < 5 && i < r; i++ {
		fmt.Printf("Row %d: [", i)
		for j := 0; j < 785 && j < c; j++ {
			fmt.Printf("%.2f ", m.At(i, j))
		}
		fmt.Println("... ]")
	}
	fmt.Println("]")
}

func printDigit(matrix [][]int) {
	for i := 0; i < 5 && i < len(matrix); i++ {
		for j := 1; j < 785 && j < len(matrix[i]); j++ {
			// fmt.Print(myChar[int(matrix[i][j]/(255/len(myChar)+1))])
			fmt.Print(string(asciiChars[int(matrix[i][j]/(255/len(asciiChars)+1))]))
			if j%28 == 0 {
				fmt.Println()
			}
		}
	}

}

func Zeros(dims ...int) interface{} {
	if len(dims) == 0 {
		return 0.0
	}

	totalSize := 1
	for _, dim := range dims {
		totalSize *= dim
	}

	// Generate a flat slice of random numbers
	flat := make([]float64, totalSize)
	for i := range flat {
		flat[i] = 0.0
	}

	// Reshape the flat slice according to the input dimensions
	return reshape(flat, dims)
}

func Rand(dims ...int) interface{} {
	if len(dims) == 0 {
		return rand.Float64()
	}

	totalSize := 1
	for _, dim := range dims {
		totalSize *= dim
	}

	// Generate a flat slice of random numbers
	flat := make([]float64, totalSize)
	for i := range flat {
		flat[i] = rand.Float64()
	}

	// Reshape the flat slice according to the input dimensions
	return reshape(flat, dims)
}

// func main() {
// 	m, _ := Subtract(Rand2D(3, 4), 0.5)
// 	printMatrix(m)
// }

type RandFunc func() float64

func Rand2D(dimx, dimy int) *mat.Dense {
	return Rand2DF(dimx, dimy, rand.Float64)
}
func Randn2D(dimx, dimy int) *mat.Dense {
	return Rand2DF(dimx, dimy, rand.NormFloat64)
}
func Rand2DF(dimx, dimy int, rf RandFunc) *mat.Dense {
	totalSize := dimx * dimy

	// Generate a flat slice of random numbers from standard normal distribution
	flat := make([]float64, totalSize)
	for i := range flat {
		flat[i] = rf()
	}

	// Reshape the flat slice according to the input dimensions
	return mat.NewDense(dimx, dimy, flat)
}

func Randn(dims ...int) interface{} {
	if len(dims) == 0 {
		return rand.NormFloat64()
	}

	totalSize := 1
	for _, dim := range dims {
		totalSize *= dim
	}

	// Generate a flat slice of random numbers from standard normal distribution
	flat := make([]float64, totalSize)
	for i := range flat {
		flat[i] = rand.NormFloat64()
	}

	// Reshape the flat slice according to the input dimensions
	return reshape(flat, dims)
}

// reshape takes a flat slice and reshapes it according to the given dimensions
func reshape(flat []float64, dims []int) interface{} {
	if len(dims) == 1 {
		return flat
	}

	subSize := 1
	for _, dim := range dims[1:] {
		subSize *= dim
	}

	result := make([]interface{}, dims[0])
	for i := range result {
		subSlice := flat[i*subSize : (i+1)*subSize]
		result[i] = reshape(subSlice, dims[1:])
	}

	return result
}

func reshape1D(flat []float64, dim int) []float64 {
	if dim == 1 {
		return flat
	}

	result := make([]float64, dim)
	for i := range result {
		subSlice := flat[i]
		result[i] = subSlice
	}

	return result
}

func Dot(a, b *mat.Dense) *mat.Dense {
	aRows, _ := a.Dims()
	bRows, bCols := b.Dims()

	result := make([]float64, aRows, bCols)

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			for k := 0; k < bRows; k++ {
				result[i*j] += a.At(i, j) * b.At(k, j)
			}
		}
	}

	return mat.NewDense(aRows, bCols, result)
}

func Maximum(a, b interface{}) (interface{}, error) {
	switch a := a.(type) {
	case []float64:
		return maximum1D(a, b)
	case [][]float64:
		return maximum2D(a, b)
	default:
		return nil, errors.New("unsupported type: only 1D and 2D slices of float64 are supported")
	}
}

func maximum1D(a []float64, b interface{}) ([]float64, error) {
	var bSlice []float64
	switch b := b.(type) {
	case float64:
		bSlice = make([]float64, len(a))
		for i := range bSlice {
			bSlice[i] = b
		}
	case []float64:
		if len(a) != len(b) {
			return nil, errors.New("arrays must have the same shape")
		}
		bSlice = b
	default:
		return nil, errors.New("unsupported type for second argument")
	}

	result := make([]float64, len(a))
	for i := range a {
		if a[i] > bSlice[i] {
			result[i] = a[i]
		} else {
			result[i] = bSlice[i]
		}
	}
	return result, nil
}

func maximum2D(a [][]float64, b interface{}) ([][]float64, error) {
	var bSlice [][]float64
	switch b := b.(type) {
	case float64:
		bSlice = make([][]float64, len(a))
		for i := range bSlice {
			bSlice[i] = make([]float64, len(a[i]))
			for j := range bSlice[i] {
				bSlice[i][j] = b
			}
		}
	case [][]float64:
		if len(a) != len(b) || len(a[0]) != len(b[0]) {
			return nil, errors.New("arrays must have the same shape")
		}
		bSlice = b
	default:
		return nil, errors.New("unsupported type for second argument")
	}

	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			if a[i][j] > bSlice[i][j] {
				result[i][j] = a[i][j]
			} else {
				result[i][j] = bSlice[i][j]
			}
		}
	}
	return result, nil
}

// Operation is a function type for element-wise operations
type Operation func(a, b float64) float64

// applyOperation is a generic function to apply an operation with broadcasting
func applyOperation(a *mat.Dense, b interface{}, op Operation) *mat.Dense {
	rows, cols := a.Dims()

	result := make([]float64, rows*cols)
	var resultMatrix *mat.Dense
	switch b := b.(type) {
	case mat.Matrix:
		resultMatrix = mat.NewDense(rows, cols, nil)
		resultMatrix.Sub(a, b.(mat.Matrix))

	case []float64:
		if len(b) != cols {
			log.Fatal("vector length must match matrix column count")
		}
		for i := range rows {
			for j := range cols {
				result[i*j] = op(a.At(i, j), b[j])
			}
		}
		resultMatrix = mat.NewDense(rows, cols, result)

	case float64:
		for i := range rows {
			for j := range cols {
				result[i*j] = op(a.At(i, j), b)
			}
		}
		resultMatrix = mat.NewDense(rows, cols, result)

	default:
		log.Fatal("unsupported type for operation")
	}

	return resultMatrix
}

// Add performs element-wise addition
func Add(a *mat.Dense, b interface{}) *mat.Dense {
	return applyOperation(a, b, func(a, b float64) float64 { return a + b })
}

// Subtract performs element-wise subtraction
func Subtract(a *mat.Dense, b interface{}) *mat.Dense {
	return applyOperation(a, b, func(a, b float64) float64 { return a - b })
}

// Multiply performs element-wise multiplication
func Multiply(a *mat.Dense, b interface{}) *mat.Dense {
	return applyOperation(a, b, func(a, b float64) float64 { return a * b })
}

// Divide performs element-wise division
func Divide(a *mat.Dense, b interface{}) *mat.Dense {
	return applyOperation(a, b, func(a, b float64) float64 {
		if b == 0 {
			return 0 // Or you could return NaN: math.NaN()
		}
		return a / b
	})
}

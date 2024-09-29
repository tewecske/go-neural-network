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
)

type Matrix [][]float64

func main() {
	// Open the CSV file
	file, err := os.Open("data/train.csv")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Create a new CSV reader
	reader := csv.NewReader(file)

	// Skip the header row
	_, err = reader.Read()
	if err != nil {
		fmt.Println("Error reading header:", err)
		return
	}

	// Initialize the matrix
	var matrix [][]int

	// Read the data rows
	for {
		record, err := reader.Read()
		if err != nil {
			// Check if we've reached the end of the file
			if err.Error() == "EOF" {
				break
			}
			fmt.Println("Error reading row:", err)
			return
		}

		// Convert the row to integers
		row := make([]int, len(record))
		for i, value := range record {
			num, err := strconv.Atoi(value)
			if err != nil {
				fmt.Printf("Error converting value to int: %s", err)
				return
			}
			row[i] = num
		}

		// Append the row to the matrix
		matrix = append(matrix, row)
	}

	m := len(matrix)
	n := len(matrix[0])
	printMatrix(matrix)
	println(m, n)

	data_train := transposeMatrix(matrix)
	printMatrix(data_train)

	Y_train := data_train[0]
	X_train := data_train[1:]

	fmt.Println(len(Y_train))
	fmt.Println(Y_train[0:10])
	fmt.Println(len(X_train))

	// W1, b1, W2, b2 := initParams()

}

func initParams() (W1 Matrix, b1 Matrix, W2 Matrix, b2 Matrix) {
	W1, err := Subtract(Randn(10, 784).(Matrix), 0.5)
	if err != nil {
		log.Fatal(err)
	}
	b1, err = Subtract(Randn(10, 1).(Matrix), 0.5)
	if err != nil {
		log.Fatal(err)
	}
	W2, err = Subtract(Randn(10, 10).(Matrix), 0.5)
	if err != nil {
		log.Fatal(err)
	}
	b2, err = Subtract(Randn(10, 1).(Matrix), 0.5)
	if err != nil {
		log.Fatal(err)
	}

	return W1, b1, W2, b2
}

func ReLU(Z [][]float64) Matrix {
	m, err := Maximum(Z, 0)
	if err != nil {
		log.Fatal(err)
	}
	return m.(Matrix)
}

func ReLUDeriv(Z Matrix) {
	return Z > 0
}

func softmax(Z Matrix) Matrix {
	m1 := Exp(Z)
	s := Sum(m1)
	A, err := Divide(m1, s)
	if err != nil {
		log.Fatal(err)
	}
	return A
}

func forwardPropagation(W1 Matrix, b1 Matrix, W2 Matrix, b2 Matrix, X Matrix) (Matrix, Matrix, Matrix, Matrix) {
	Z1, err := Dot(W1, X)
	if err != nil {
		log.Fatal(err)
	}
	Z1, err = Add(Z1, b1)
	if err != nil {
		log.Fatal(err)
	}
	A1 := ReLU(Z1)
	Z2, err := Dot(W2, A1)
	if err != nil {
		log.Fatal(err)
	}
	Z2, err = Add(Z2, b2)
	if err != nil {
		log.Fatal(err)
	}
	A2 := softmax(A1)

	return Z1, A1, Z2, A2
}

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

func backPropagation(Z1 Matrix, A1 Matrix, Z2 Matrix, A2 Matrix, W2 Matrix, Y []int, m int) {
	oneHotY := oneHot(Y)
	dZ2, err := Subtract(A2, oneHotY)
	if err != nil {
		log.Fatal(err)
	}
	dZ2dotA1T, err := Dot(dZ2, transposeMatrix(A1))
	if err != nil {
		log.Fatal(err)
	}
	dW2, err := Multiply(dZ2dotA1T, 1/m)
	if err != nil {
		log.Fatal(err)
	}
	db2 := 1 / float64(m) * npSum(dZ2)
}

func npSum(a Matrix) float64 {
	total := 0.0
	for _, row := range a {
		for _, val := range row {
			total += val
		}
	}
	return total
}

func SumA(a []float64) float64 {
	sum := 0.0
	for _, n := range a {
		sum += n
	}
	return sum
}

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

func printMatrix(matrix [][]int) {
	// Print the dimensions of the matrix
	fmt.Printf("Matrix dimensions: %d rows x %d columns", len(matrix), len(matrix[0]))

	// Example: Print the first 5 rows and 10 columns
	for i := 0; i < 5 && i < len(matrix); i++ {
		fmt.Printf("Row %d: ", i)
		for j := 0; j < 10 && j < len(matrix[i]); j++ {
			fmt.Printf("%3d ", matrix[i][j])
		}
		fmt.Println("...")
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

func Dot(a, b Matrix) (Matrix, error) {
	if len(a) == 0 || len(b) == 0 || len(a[0]) == 0 || len(b[0]) == 0 {
		return nil, errors.New("empty matrix")
	}
	if len(a[0]) != len(b) {
		return nil, errors.New("incompatible matrix dimensions")
	}

	rows, cols := len(a), len(b[0])
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for k := 0; k < len(b); k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return result, nil
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
func applyOperation(a Matrix, b interface{}, op Operation) (Matrix, error) {
	if len(a) == 0 || len(a[0]) == 0 {
		return nil, errors.New("empty matrix")
	}

	rows, cols := len(a), len(a[0])
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	switch b := b.(type) {
	case Matrix:
		if len(b) != rows || len(b[0]) != cols {
			return nil, errors.New("matrices must have the same dimensions")
		}
		for i := range result {
			for j := range result[i] {
				result[i][j] = op(a[i][j], b[i][j])
			}
		}

	case []float64:
		if len(b) != cols {
			return nil, errors.New("vector length must match matrix column count")
		}
		for i := range result {
			for j := range result[i] {
				result[i][j] = op(a[i][j], b[j])
			}
		}

	case float64:
		for i := range result {
			for j := range result[i] {
				result[i][j] = op(a[i][j], b)
			}
		}

	default:
		return nil, errors.New("unsupported type for operation")
	}

	return result, nil
}

// Add performs element-wise addition
func Add(a Matrix, b interface{}) (Matrix, error) {
	return applyOperation(a, b, func(a, b float64) float64 { return a + b })
}

// Subtract performs element-wise subtraction
func Subtract(a Matrix, b interface{}) (Matrix, error) {
	return applyOperation(a, b, func(a, b float64) float64 { return a - b })
}

// Multiply performs element-wise multiplication
func Multiply(a Matrix, b interface{}) (Matrix, error) {
	return applyOperation(a, b, func(a, b float64) float64 { return a * b })
}

// Divide performs element-wise division
func Divide(a Matrix, b interface{}) (Matrix, error) {
	return applyOperation(a, b, func(a, b float64) float64 {
		if b == 0 {
			return 0 // Or you could return NaN: math.NaN()
		}
		return a / b
	})
}

type UnaryOperation func(a float64) float64

func applyUnaryOperation(a Matrix, op UnaryOperation) Matrix {
	result := make(Matrix, len(a))
	for i, row := range a {
		result[i] = make([]float64, len(row))
		for j, val := range row {
			result[i][j] = op(val)
		}
	}
	return result
}

// Exp performs element-wise exponential operation
func Exp(a Matrix) Matrix {
	return applyUnaryOperation(a, math.Exp)
}

func Rec(a Matrix) Matrix {
	return applyUnaryOperation(a, func(x float64) float64 {
		return 1 / x
	})
}

func Sum(a Matrix) []float64 {
	if len(a) == 0 || len(a[0]) == 0 {
		return []float64{}
	}

	cols := len(a[0])
	result := make([]float64, cols)

	for _, row := range a {
		for j, val := range row {
			result[j] += val
		}
	}

	return result
}

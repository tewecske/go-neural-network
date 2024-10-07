package main

import t "gorgonia.org/tensor"

type LayerDense struct {
	Weights t.Tensor
	Biases  []float64
	Output  t.Tensor
}

func NewLayerDense(nInputs, nNeurons int) LayerDense {
	weights := t.New(t.WithShape(nInputs, nNeurons), t.WithBacking(t.Random(t.Float64, nInputs*nNeurons)))
	weights = unsafe(weights.MulScalar(0.01, false))

	biases := make([]float64, nNeurons)

	return LayerDense{
		Weights: weights,
		Biases:  biases,
	}
}

func (l *LayerDense) Forward(inputs t.Tensor) {
	dp := unsafe(t.Dot(inputs, l.Weights))
	if len(dp.Shape()) == 1 {
		dp.Reshape(dp.Shape()[0], 1)
	}
	l.Output = unsafe(AddVector(dp, l.Biases))
}

package main

import (
	"math"

	t "gorgonia.org/tensor"
)

type ActivationReLU struct{}

func NewActivationReLU() ActivationReLU {
	return ActivationReLU{}
}

func (a *ActivationReLU) Forward(l LayerDense) t.Tensor {
	outputs := make([]float64, l.Output.Shape()[0])
	for i := range l.Output.Shape()[0] {
		outputs[i] = math.Max(0, unsafe(l.Output.At(i, 0)).(float64))
	}
	return t.New(t.WithShape(l.Output.Shape()...), t.WithBacking(outputs))
}

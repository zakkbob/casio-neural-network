package internal

import "math/rand"

func preserve(n float64) float64 {
	return n
}

func relu(n float64) float64 {
	return max(0, n)
}

type node struct {
	processActivation func(float64) float64
	weights           []float64
	bias              float64
}

func randomNode(in int, activationFn func(float64) float64) node {
	n := node{
		processActivation: activationFn,
		weights:           make([]float64, in),
		bias:              rand.Float64(),
	}

	for i := range in {
		n.weights[i] = rand.Float64()*2 - 1
	}

	return n
}

func (n *node) activate(in []float64) float64 {
	if len(in) != len(n.weights) {
		panic("node's input value differ in length from it's weights")
	}

	sum := n.bias

	for i := range in {
		sum += in[i] * n.weights[i]
	}

	return n.processActivation(sum)
}

type layer struct {
	nodes []node
}

func randomLayer(in, out int, activationFn func(float64) float64) layer {
	l := layer{
		nodes: make([]node, out),
	}

	for i := range out {
		l.nodes[i] = randomNode(in, activationFn)
	}

	return l
}

func (l *layer) process(in []float64) []float64 {
	out := make([]float64, len(l.nodes))

	for i, n := range l.nodes {
		out[i] = n.activate(in)
	}

	return out
}

type network struct {
	inputLen int // length of input layer
	layers   []layer
}

func (n *network) process(in []float64) []float64 {
	for _, layer := range n.layers {
		in = layer.process(in)
	}

	return in
}

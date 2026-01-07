package internal

func relu(n float64) float64 {
	return max(0, n)
}

type node struct {
	processActivation func(float64) float64
	weights           []float64
	bias              float64
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

func (l *layer) process(in []float64) []float64 {
	if len(in) != len(l.nodes) {
		panic("layer's input value differs in length from its nodes")
	}

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

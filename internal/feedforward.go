package internal

import (
	"math/rand"
	"slices"
	"fmt"
)

func identity(n float64) float64 {
	return n
}

func relu(n float64) float64 {
	return max(0, n)
}

type node struct {
	weights []float64
	bias    float64
}

func randomNode(in int, activation func(float64) float64) node {
	n := node{
		weights: make([]float64, in),
		bias:    rand.Float64(),
	}

	for i := range in {
		n.weights[i] = rand.Float64()
	}

	return n
}

func (n *node) predict(in []float64) float64 {
	if len(in) != len(n.weights) {
		panic("node's input value differ in length from it's weights")
	}

	sum := n.bias

	for i := range in {
		sum += in[i] * n.weights[i]
	}

	return relu(sum)
}

type layer struct {
	nodes       []node
	lastOutputs []float64
}

func (l *layer) clone() *layer {
	return &layer{
		nodes:       slices.Clone(l.nodes),
		lastOutputs: slices.Clone(l.lastOutputs),
	}
}

func randomLayer(in, out int, activationFn func(float64) float64) *layer {
	l := layer{
		nodes: make([]node, out),
	}

	for i := range out {
		l.nodes[i] = randomNode(in, activationFn)
	}

	return &l
}

func (l *layer) predict(in []float64) []float64 {
	out := make([]float64, len(l.nodes))

	for i, n := range l.nodes {
		out[i] = n.predict(in)
	}

	l.lastOutputs = out

	return out
}

type network struct {
	inputLen int // length of input layer
	layers   []*layer
}

func (n *network) predict(in []float64) []float64 {
	for _, layer := range n.layers {
		in = layer.predict(in)
	}

	return in
}

func (n *network) clone() network { // yeah i really need to switch to matrix multiplication
	layers := make([]*layer, len(n.layers))

	for i, layer := range n.layers {
		layers[i] = layer.clone()
	}

	return network{
		n.inputLen,
		layers,
	}
}

func (n *network) train(ins [][]float64, outs [][]float64, passes int) {
	var alpha float64 = 0.1

	for range passes {
		old := n.clone()

		losses := []float64{}

		for i, in := range ins {
			out := outs[i]

			old.predict(in)

			lastLosses := []float64{}

			outputLayer := old.layers[len(n.layers)-1]
			for j := range outputLayer.nodes { // output layer
				output := outputLayer.lastOutputs[j]

				loss := output - out[j]
				lastLosses = append(lastLosses, loss)

if output <= 0 { // ReLU dead thing
					continue
				}

				if len(n.layers) > 1 {
					for k := range n.layers[len(n.layers)-2].nodes {
						gradient := old.layers[len(n.layers)-2].lastOutputs[k] * loss

						n.layers[len(n.layers)-1].nodes[j].weights[k] += -alpha * gradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
					}
				} else {
					for k := range n.inputLen {
						gradient := in[k] * loss

						n.layers[len(n.layers)-1].nodes[j].weights[k] += -alpha * gradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
					}
				}

				biasGradient := outputLayer.nodes[j].bias * loss
				n.layers[len(n.layers)-1].nodes[j].bias += -alpha * biasGradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
			}

			for l := len(n.layers) - 2; l >= 0; l-- { // hidden layers
				layer := old.layers[l]
				losses = []float64{}

				for j := range layer.nodes { // output layer
					output := layer.lastOutputs[j]

					if output <= 0 { // ReLU dead thing
						continue
					}

					fmt.Println(lastLosses)
					loss := output
					for o := range n.layers[l+1].nodes {
						loss -= n.layers[l+1].nodes[o].weights[j] * lastLosses[o]
					}

					losses = append(losses, loss)

					if l != 0 {
						for k := range n.layers[l-1].nodes {
							gradient := old.layers[l-1].lastOutputs[k] * loss

							n.layers[l].nodes[j].weights[k] += -alpha * gradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
						}
					} else {
						for k := range n.inputLen {
							gradient := in[k] * loss

							n.layers[l].nodes[j].weights[k] += -alpha * gradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
						}
					}

					biasGradient := layer.nodes[j].bias * loss
					n.layers[l].nodes[j].bias += -alpha * biasGradient / float64(len(ins)) // divide by the length of ins, removes need to sum and calculate mean after. probably increases floating point loss
				}

				lastLosses = losses
			}
		}
	}
}

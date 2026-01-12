package internal

import (
	"slices"
	"testing"
)

func TestNode(t *testing.T) {
	n := node{
		weights: []float64{1, 2, 3},
		bias:    4,
	}

	out := n.predict([]float64{3, 1, 4})

	if out != 21 {
		t.Errorf("expected 21, got %v", out)
	}
}

func TestNetwork(t *testing.T) {
	n := network{
		inputLen: 2,
		layers: []*layer{
			{
				nodes: []node{
					{
						weights: []float64{1, 2},
						bias:    4,
					},
					{
						weights: []float64{0.5, 0.25},
						bias:    2,
					},
					{
						weights: []float64{1, 2},
						bias:    1,
					},
				},
			},
			{
				nodes: []node{
					{
						weights: []float64{2, 4, 0.5},
						bias:    4,
					},
					{
						weights: []float64{3, -0.5, 2},
						bias:    3,
					},
				},
			},
		},
	}

	out := n.predict([]float64{-1, 4})

	if !slices.Equal(out, []float64{-27, -9.25}) {
		t.Errorf("expected [-27, -9.25], got %v", out)
	}
}

func TestTrain(t *testing.T) {
	n := network{
		inputLen: 1,
		layers: []*layer{
			randomLayer(1, 2, relu),
		},
	}

	n.train([][]float64{
		{0},
		{0.1},
		{0.5},
		{0.7},
	}, [][]float64{
		{0, 0.1}, // n, 0.5n+0.1
		{0.1, 0.15},
		{0.5, 0.35},
		{0.7, 0.45},
	}, 1000)

	out := n.predict([]float64{0.3})

	if !(out[0]-0.3 < 0.1 && out[1]-0.25 < 0.1) {
		t.Errorf("expected close to [0.3, 0.25], got %v", out)
	}
}

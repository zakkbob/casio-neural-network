package internal

import (
	"slices"
	"testing"
)

func TestNode(t *testing.T) {
	n := node{
		processActivation: preserve,
		weights:           []float64{1, 2, 3},
		bias:              4,
	}

	out := n.activate([]float64{3, 1, 4})

	if out != 21 {
		t.Errorf("expected 21, got %v", out)
	}
}

func TestNetwork(t *testing.T) {
	n := network{
		inputLen: 2,
		layers: []layer{
			{
				nodes: []node{
					{
						processActivation: preserve,
						weights:           []float64{1, 2},
						bias:              4,
					},
					{
						processActivation: preserve,
						weights:           []float64{0.5, 0.25},
						bias:              -2,
					},
					{
						processActivation: preserve,
						weights:           []float64{-1, 2},
						bias:              1,
					},
				},
			},
			{
				nodes: []node{
					{
						processActivation: preserve,
						weights:           []float64{-2, 4, 0.5},
						bias:              -4,
					},
					{
						processActivation: preserve,
						weights:           []float64{-3, -0.5, 2},
						bias:              3,
					},
				},
			},
		},
	}

	out := n.process([]float64{-1, 4})

	if !slices.Equal(out, []float64{-27, -9.25}) {
		t.Errorf("expected [-27, -9.25], got %v", out)
	}
}

package internal

import "testing"

func TestNode(t *testing.T) {
	n := node{
		processActivation: relu,
		weights:           []float64{1, 2, 3},
		bias:              4,
	}

	out := n.activate([]float64{3, 1, 4})

	if out != 21 {
		t.Errorf("expected 21, got %v", out)
	}
}

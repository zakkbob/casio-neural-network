package internal

import (
	"testing"
)

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
	}, 10000)

	out := n.predict([]float64{0.3})

	t.Log(out)

	if !(out[0]-0.3 < 0.1 && out[1]-0.25 < 0.1) {
		t.Errorf("expected close to [0.3, 0.25], got %v", out)
	}
}

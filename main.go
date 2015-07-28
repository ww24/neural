package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/ww24/neural/network"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	nw := network.New(
		network.Option{
			LayerSizes: []int{1, 3, 1}})

	eta := 0.1

	for i := 0; i <= 100000; i++ {
		for j := 0; j < 100; j++ {
			input := math.Pi * 2 * float64(j) / 100
			output := math.Sin(input)
			nw.Train(eta, []float64{input}, []float64{output})
		}

		if i%10000 == 0 {
			a := nw.Ignite([]float64{math.Pi * 0})
			b := nw.Ignite([]float64{math.Pi * 0.5})
			c := nw.Ignite([]float64{math.Pi * 1})
			fmt.Printf("train:%7d x=%f, %f; x=%f, %f; x=%f, %f\n", i, 0.0, a[0], math.Pi*0.5, b[0], math.Pi, c[0])
		}
	}
}

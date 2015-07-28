Neural Network
==============

Usage
-----
```
rand.Seed(time.Now().UnixNano())

nw := network.New(
	network.Option{
		LayerSizes: []int{2, 2, 1}})

// train
eta := 0.1
inputs := []float64{0,1}
trainingData := []float64{0}
nw.Train(eta, inputs, trainingData)
results := nw.Train(eta, inputs, trainingData)

// calculate
results := nw.Ignite([]float64{1, 0}, outputs)
```

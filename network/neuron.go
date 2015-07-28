/**
 * neuron
 */

package network

import (
	"errors"
	"math"
	"math/rand"
	"strconv"
)

// Network structure
type Network struct {
	Layers []Layer
}

// Layer structure
type Layer struct {
	Neurons []Neuron
}

// Neuron structure
type Neuron struct {
	Threshold float64
	Weights   []float64
}

// Option is Network constructor option
type Option struct {
	LayerSizes []int
}

// New is Network constructor
func New(option Option) (network Network) {
	network.Layers = make([]Layer, len(option.LayerSizes)-1)

	for index, size := range option.LayerSizes {
		if index == 0 {
			// 入力層ではニューロンを生成しない
			continue
		}

		// 上位層のニューロンの数と現在の層のニューロンの数を渡して層の初期化
		network.Layers[index-1] = newLayer(option.LayerSizes[index-1], size)
	}

	return
}

func newLayer(topLayerSize int, currentLayerSize int) (layer Layer) {
	layer.Neurons = make([]Neuron, currentLayerSize)

	// 現在の層のニューロンの数だけニューロンを生成
	for i := 0; i < currentLayerSize; i++ {
		// 上位層のニューロンの数を渡してニューロンの初期化
		layer.Neurons[i] = newNewron(topLayerSize)
	}

	return
}

func newNewron(topLayerSize int) (neuron Neuron) {
	neuron.Threshold = rand.Float64() / 1000
	neuron.Weights = make([]float64, topLayerSize)
	for i := range neuron.Weights {
		neuron.Weights[i] = rand.Float64() / 1000
	}

	return
}

// Ignite is calculate method
func (network *Network) Ignite(inputs []float64, ref ...[][]float64) (results []float64) {
	var outputList [][]float64

	// 省略可能な可変長引数の先頭だけを受け取る
	if len(ref) > 0 {
		outputList = ref[0]
	}

	// 入力値エラーハンドリング
	if len(inputs) != len(network.Layers[0].Neurons[0].Weights) {
		inputSize := len(network.Layers[0].Neurons[0].Weights)
		err := errors.New("inputs must have " + strconv.Itoa(inputSize) + " items.")
		panic(err)
	}

	// ignite layer
	for i, layer := range network.Layers {
		inputs = layer.ignite(inputs)
		if outputList != nil {
			outputList[i] = inputs
		}
	}

	results = inputs
	return
}

func (layer *Layer) ignite(inputs []float64) (results []float64) {
	results = make([]float64, len(layer.Neurons))

	// ignite neuron
	for index, neuron := range layer.Neurons {
		results[index] = neuron.ignite(inputs)
	}

	return
}

func (neuron *Neuron) ignite(inputs []float64) (result float64) {
	// 入力 * 重みの総和を求める
	for index, weight := range neuron.Weights {
		result += inputs[index] * weight
	}

	// 入力 * 重みの総和から閾値を引いた値をシグモイド関数に渡す
	result = sigmoid(result - neuron.Threshold)

	return
}

func sigmoid(input float64) (result float64) {
	result = 1 / (1 + math.Exp(-input))
	return
}

// Train is training method (eta: 学習定数)
func (network *Network) Train(eta float64, inputs []float64, trainingData []float64) {
	// 各層の出力を保持する slice を用意して発火
	outputList2 := make([][]float64, len(network.Layers))
	network.Ignite(inputs, outputList2)

	// prepend inputs []float64 to outputList [][]float64
	outputList2 = append([][]float64{inputs}, outputList2...)

	// 最も neuron が多い層の neuron の数で slice を確保
	maxLayerSize := float64(0)
	for _, layer := range network.Layers {
		maxLayerSize = math.Max(maxLayerSize, float64(len(layer.Neurons)))
	}
	epsilons := make([]float64, int(maxLayerSize))

	// 各層の各 neuron の数だけループ
	for i := len(network.Layers) - 1; i >= 0; i-- {
		layer := network.Layers[i]
		outputList1 := outputList2[i+1]

		for j, neuron := range layer.Neurons {
			epsilons[j] = outputList1[j] * (1 - outputList1[j])

			if i == len(network.Layers)-1 {
				// 出力層の誤差
				epsilons[j] *= (outputList1[j] - trainingData[j])
			} else {
				// 中間層の誤差
				e := float64(0)
				for index, neuron := range network.Layers[i+1].Neurons {
					e += epsilons[j] * neuron.Weights[index]
				}
				epsilons[j] *= e
			}

			// 重みと閾値の更新
			for index := range append(neuron.Weights, neuron.Threshold) {
				delta := -eta * epsilons[j]

				if index < len(neuron.Weights) {
					// 重みの更新
					neuron.Weights[index] += delta * outputList2[i][index]
				} else {
					// 閾値の更新 (入力は -1 固定)
					neuron.Threshold += delta * -1
				}
			}
		}
	}
}

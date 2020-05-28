package randomforest

//a random forest implemtation in GoLang
import (
	"math"
	"math/rand"
)

const CAT = "cat"
const NUMERIC = "numeric"

type TreeNode struct {
	ColumnNo int //column number
	Value    interface{}
	Left     *TreeNode
	Right    *TreeNode
	Labels   map[string]int
}

type Tree struct {
	Root *TreeNode
}

func getRandomRange(N int, M int) []int {
	tmp := make([]int, N)
	for i := 0; i < N; i++ {
		tmp[i] = i
	}
	for i := 0; i < M; i++ {
		j := i + int(rand.Float64()*float64(N-i))
		tmp[i], tmp[j] = tmp[j], tmp[i]
	}

	return tmp[:M]
}

func getSamples(ary [][]interface{}, index []int) [][]interface{} {
	result := make([][]interface{}, len(index))
	for i := 0; i < len(index); i++ {
		result[i] = ary[index[i]]
	}
	return result
}

func getLabels(ary []string, index []int) []string {
	result := make([]string, len(index))
	for i := 0; i < len(index); i++ {
		result[i] = ary[index[i]]
	}
	return result
}

func getEntropy(epMap map[string]float64, total int) float64 {

	for k := range epMap {
		epMap[k] = epMap[k] / float64(total) //normalize
	}

	entropy := 0.0
	for _, v := range epMap {
		entropy += v * math.Log(1.0/v)
	}

	return entropy
}

func getGini(epMap map[string]float64) float64 {
	total := 0.0
	for _, v := range epMap {
		total += v
	}

	for k := range epMap {
		epMap[k] = epMap[k] / total //normalize
	}

	impure := 0.0
	for k1, v1 := range epMap {
		for k2, v2 := range epMap {
			if k1 != k2 {
				impure += v1 * v2
			}
		}
	}
	return impure
}

func getBestGain(samples [][]interface{}, c int, sampleLabels []string, columnType string, currentEntropy float64) (float64, interface{}, int, int) {
	var bestValue interface{}
	bestGain := 0.0
	bestTotalR := 0
	bestTotalL := 0

	uniqueValues := make(map[interface{}]int)
	for i := 0; i < len(samples); i++ {
		uniqueValues[samples[i][c]] = 1
	}

	for value := range uniqueValues {
		mapL := make(map[string]float64)
		mapR := make(map[string]float64)
		totalL := 0
		totalR := 0
		if columnType == CAT {
			for j := 0; j < len(samples); j++ {
				if samples[j][c] == value {
					totalL++
					mapL[sampleLabels[j]] += 1.0
				} else {
					totalR++
					mapR[sampleLabels[j]] += 1.0
				}
			}
		}
		if columnType == NUMERIC {
			for j := 0; j < len(samples); j++ {
				if samples[j][c].(float64) <= value.(float64) {
					totalL++
					mapL[sampleLabels[j]] += 1.0
				} else {
					totalR++
					mapR[sampleLabels[j]] += 1.0
				}
			}
		}

		p1 := float64(totalR) / float64(len(samples))
		p2 := float64(totalL) / float64(len(samples))

		newEntropy := p1*getEntropy(mapR, totalR) + p2*getEntropy(mapL, totalL)
		//fmt.Println(newEntropy,currentEntropy)
		entropyGain := currentEntropy - newEntropy

		if entropyGain >= bestGain {
			bestGain = entropyGain
			bestValue = value
			bestTotalL = totalL
			bestTotalR = totalR
		}
	}

	return bestGain, bestValue, bestTotalL, bestTotalR
}

func splitSamples(samples [][]interface{}, columnType string, c int, value interface{}, part_l *[]int, part_r *[]int) {
	if columnType == CAT {
		for j := 0; j < len(samples); j++ {
			if samples[j][c] == value {
				*part_l = append(*part_l, j)
			} else {
				*part_r = append(*part_r, j)
			}
		}
	}
	if columnType == NUMERIC {
		for j := 0; j < len(samples); j++ {
			if samples[j][c].(float64) <= value.(float64) {
				*part_l = append(*part_l, j)
			} else {
				*part_r = append(*part_r, j)
			}
		}
	}
}

func buildTree(samples [][]interface{}, samples_labels []string, selectedFeatureCount int) *TreeNode {
	//fmt.Println(len(samples))
	//find a best splitter
	columnCount := len(samples[0])
	//splitCount := int(math.Log(float64(columnCount)))
	splitCount := selectedFeatureCount
	columnsChoosen := getRandomRange(columnCount, splitCount)

	bestGain := 0.0
	var bestPartL []int = make([]int, 0, len(samples))
	var bestPartR []int = make([]int, 0, len(samples))
	var bestTotalL int = 0
	var bestTotalR int = 0
	var bestValue interface{}
	var bestColumn int
	var bestColumnType string

	currentEntropyMap := make(map[string]float64)
	for i := 0; i < len(samples_labels); i++ {
		currentEntropyMap[samples_labels[i]]++
	}

	currentEntropy := getEntropy(currentEntropyMap, len(samples_labels))

	for _, c := range columnsChoosen {
		columnType := CAT
		if _, ok := samples[0][c].(float64); ok {
			columnType = NUMERIC
		}

		gain, value, totalL, totalR := getBestGain(samples, c, samples_labels, columnType, currentEntropy)
		//fmt.Println("kkkkk",gain,part_l,part_r)
		if gain >= bestGain {
			bestGain = gain
			bestValue = value
			bestColumn = c
			bestColumnType = columnType
			bestTotalL = totalL
			bestTotalR = totalR
		}
	}

	if bestGain > 0 && bestTotalL > 0 && bestTotalR > 0 {
		node := &TreeNode{}
		node.Value = bestValue
		node.ColumnNo = bestColumn
		splitSamples(samples, bestColumnType, bestColumn, bestValue, &bestPartL, &bestPartR)
		node.Left = buildTree(getSamples(samples, bestPartL), getLabels(samples_labels, bestPartL), selectedFeatureCount)
		node.Right = buildTree(getSamples(samples, bestPartR), getLabels(samples_labels, bestPartR), selectedFeatureCount)
		return node
	}

	return genLeafNode(samples_labels)

}

func genLeafNode(labels []string) *TreeNode {
	counter := make(map[string]int)
	for _, v := range labels {
		counter[v]++
	}

	node := &TreeNode{}
	node.Labels = counter
	//fmt.Println(node)
	return node
}

func predict(node *TreeNode, input []interface{}) map[string]int {
	if node.Labels != nil { //leaf node
		return node.Labels
	}

	c := node.ColumnNo
	value := input[c]

	switch value.(type) {
	case float64:
		if value.(float64) <= node.Value.(float64) && node.Left != nil {
			return predict(node.Left, input)
		} else if node.Right != nil {
			return predict(node.Right, input)
		}
	case string:
		if value == node.Value && node.Left != nil {
			return predict(node.Left, input)
		} else if node.Right != nil {
			return predict(node.Right, input)
		}
	}

	return nil
}

func BuildTree(inputs [][]interface{}, labels []string, samplesCount, selectedFeatureCount int) *Tree {

	samples := make([][]interface{}, samplesCount)
	sampleLabels := make([]string, samplesCount)
	for i := 0; i < samplesCount; i++ {
		j := int(rand.Float64() * float64(len(inputs)))
		samples[i] = inputs[j]
		sampleLabels[i] = labels[j]
	}

	tree := &Tree{}
	tree.Root = buildTree(samples, sampleLabels, selectedFeatureCount)

	return tree
}

func PredictTree(tree *Tree, input []interface{}) map[string]int {
	return predict(tree.Root, input)
}

package randomforest

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

type Forest struct {
	Trees []*Tree
}

func BuildForest(inputs [][]interface{}, labels []string, treesAmount, samplesAmount, selectedFeatureAmount int) *Forest {
	rand.Seed(time.Now().UnixNano())
	forest := &Forest{}
	forest.Trees = make([]*Tree, treesAmount)
	doneFlag := make(chan bool)
	progCounter := 0
	mutex := &sync.Mutex{}
	for i := 0; i < treesAmount; i++ {
		go func(x int) {
			fmt.Printf(">> %v buiding %vth tree...\n", time.Now(), x)
			forest.Trees[x] = BuildTree(inputs, labels, samplesAmount, selectedFeatureAmount)
			//fmt.Printf("<< %v the %vth tree is done.\n",time.Now(), x)
			mutex.Lock()
			progCounter++
			fmt.Printf("%v tranning progress %.0f%%\n", time.Now(), float64(progCounter)/float64(treesAmount)*100)
			mutex.Unlock()
			doneFlag <- true
		}(i)
	}

	for i := 1; i <= treesAmount; i++ {
		<-doneFlag
	}

	fmt.Println("all done.")
	return forest
}

func DefaultForest(inputs [][]interface{}, labels []string, treesAmount int) *Forest {
	m := int(math.Sqrt(float64(len(inputs[0]))))
	n := int(math.Sqrt(float64(len(inputs))))
	return BuildForest(inputs, labels, treesAmount, n, m)
}

func (forest *Forest) Predict(input []interface{}) string {
	counter := make(map[string]float64)
	for i := 0; i < len(forest.Trees); i++ {
		treeCounter := PredictTree(forest.Trees[i], input)
		total := 0.0
		for _, v := range treeCounter {
			total += float64(v)
		}
		for k, v := range treeCounter {
			counter[k] += float64(v) / total
		}
	}

	maxValue := 0.0
	maxLabel := ""
	for k, v := range counter {
		if v >= maxValue {
			maxValue = v
			maxLabel = k
		}
	}
	return maxLabel
}

// PredictStable takes the most common guess of all the trees. If several labels have the same amount of guesses
// it chooses the label that's alphabetically first
func (forest *Forest) PredictStable(input []interface{}) string {
	counter := make(map[string]int)
	for _, tree := range forest.Trees {
		treeCounter := PredictTree(tree, input)
		maxKey := maxKey(treeCounter)
		counter[maxKey]++
	}
	return maxKey(counter)
}

func maxKey(kvmap map[string]int) string {
	var maxKey string
	var maxVal int
	for k, v := range kvmap {
		if v > maxVal || (v == maxVal && strings.Compare(k, maxKey) < 0) {
			maxKey = k
			maxVal = v
		}
	}
	return maxKey
}

func DumpForest(forest *Forest, fileName string) {
	file, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to create " + fileName)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.Encode(forest)
}

func LoadForest(fileName string) *Forest {
	file, err := os.Open(fileName)
	if err != nil {
		panic("failed to open " + fileName)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	forest := &Forest{}
	decoder.Decode(forest)
	return forest
}

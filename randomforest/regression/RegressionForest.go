package regression

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

type Forest struct {
	Trees []*Tree
}

func BuildForest(inputs [][]interface{}, labels []float64, treesAmount, samplesAmount, selectedFeatureAmount int) *Forest {
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

func DefaultForest(inputs [][]interface{}, labels []float64, treesAmount int) *Forest {
	m := int(math.Sqrt(float64(len(inputs[0]))))
	n := int(math.Sqrt(float64(len(inputs))))
	return BuildForest(inputs, labels, treesAmount, n, m)
}

func (self *Forest) Predicate(input []interface{}) float64 {
	total := 0.0
	for i := 0; i < len(self.Trees); i++ {
		total += PredicateTree(self.Trees[i], input)
	}
	avg := total / float64(len(self.Trees))
	return avg
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

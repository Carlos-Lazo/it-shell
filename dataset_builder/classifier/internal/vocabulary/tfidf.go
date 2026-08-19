package vocabulary

import (
	"math"
)

func TF(tokens []string) map[string]float64 {

	tf := map[string]float64{}

	for _, t := range tokens {
		tf[t]++
	}

	total := float64(len(tokens))

	for k, v := range tf {
		tf[k] = v / total
	}

	return tf

}

func IDF(documents [][]string) map[string]float64 {

	df := map[string]int{}

	N := len(documents)

	for _, doc := range documents {

		seen := map[string]bool{}

		for _, w := range doc {

			if !seen[w] {

				df[w]++
				seen[w] = true

			}

		}

	}

	idf := map[string]float64{}

	for word, count := range df {

		idf[word] = math.Log((1+float64(N))/(1+float64(count))) + 1

	}

	return idf

}

func TFIDF(tokens []string, idf map[string]float64) map[string]float64 {

	tf := TF(tokens)

	out := map[string]float64{}

	for word, value := range tf {

		out[word] = value * idf[word]

	}

	return out

}

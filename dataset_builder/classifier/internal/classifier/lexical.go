package classifier

import (
	"math"
	"sort"

	"classifier/internal/preprocess"
	"classifier/internal/vocabulary"
)

const (
	// MinScore es la similitud mínima para considerar una predicción confiable.
	MinScore = 0.12
	// MinMargin es la diferencia mínima entre las dos primeras categorías.
	MinMargin = 0.04
)

type Prediction struct {
	Category string
	Score    float64
}

// Predict clasifica texto mediante la similitud coseno con cada categoría.
func Predict(text string, model vocabulary.VocabularyModel) (string, float64) {
	predictions := PredictTopK(text, model, 1)
	if len(predictions) == 0 {
		return "", 0
	}
	return predictions[0].Category, predictions[0].Score
}

// PredictTopK devuelve las categorías coincidentes ordenadas por similitud.
func PredictTopK(text string, model vocabulary.VocabularyModel, k int) []Prediction {
	tokens := preprocess.Tokenize(text)
	if len(tokens) == 0 || k <= 0 {
		return nil
	}
	query := vocabulary.TFIDF(tokens, model.IDF)
	queryNorm := norm(query)
	if queryNorm == 0 {
		return nil
	}

	predictions := make([]Prediction, 0, len(model.Categories))
	for category, words := range model.Categories {
		score := 0.0
		for token, weight := range query {
			score += weight * words[token]
		}
		wordsNorm := norm(words)
		if wordsNorm == 0 {
			continue
		}
		score /= queryNorm * wordsNorm
		if score > 0 {
			predictions = append(predictions, Prediction{Category: category, Score: score})
		}
	}

	sort.Slice(predictions, func(i, j int) bool {
		if predictions[i].Score == predictions[j].Score {
			return predictions[i].Category < predictions[j].Category
		}
		return predictions[i].Score > predictions[j].Score
	})
	if k < len(predictions) {
		return predictions[:k]
	}
	return predictions
}

// IsInconclusive identifica resultados débiles o con dos categorías muy cercanas.
func IsInconclusive(predictions []Prediction, minScore, minMargin float64) bool {
	if len(predictions) == 0 || predictions[0].Score < minScore {
		return true
	}
	return len(predictions) > 1 && predictions[0].Score-predictions[1].Score < minMargin
}

func norm(vector map[string]float64) float64 {
	sum := 0.0
	for _, value := range vector {
		sum += value * value
	}
	return math.Sqrt(sum)
}

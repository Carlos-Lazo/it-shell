package vocabulary

import (
	"classifier/internal/preprocess"
	"classifier/internal/types"
	"encoding/json"
	"os"
	"sort"
)

type VocabularyModel struct {
	Categories map[string]map[string]float64 `json:"categories"`
	IDF        map[string]float64            `json:"idf"`
}

func BuildVocabulary(commands []types.Command) VocabularyModel {

	categoryTokens := make(map[string][]string)

	// Agrupar documentos por categoría
	for _, cmd := range commands {

		text := cmd.Command + " " + cmd.Description

		tokens := preprocess.Tokenize(text)

		categoryTokens[cmd.Category] = append(categoryTokens[cmd.Category], tokens...)

	}

	model := VocabularyModel{
		Categories: make(map[string]map[string]float64),
		IDF:        make(map[string]float64),
	}

	// Cada categoría es un documento compuesto por sus comandos y descripciones.
	// Así el IDF destaca términos que distinguen una categoría de las demás.
	categories := make([]string, 0, len(categoryTokens))
	for category := range categoryTokens {
		categories = append(categories, category)
	}
	sort.Strings(categories)
	documents := make([][]string, 0, len(categories))
	for _, category := range categories {
		documents = append(documents, categoryTokens[category])
	}
	model.IDF = IDF(documents)
	for _, category := range categories {
		model.Categories[category] = TFIDF(categoryTokens[category], model.IDF)
	}

	return model

}

func Score(tokens []string, vocabulary map[string]float64) float64 {

	score := 0.0

	for _, t := range tokens {

		score += vocabulary[t]

	}

	return score

}

func Save(model VocabularyModel, path string) error {

	data, err := json.MarshalIndent(model, "", "  ")

	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)

}

func Load(path string) (VocabularyModel, error) {

	var model VocabularyModel

	data, err := os.ReadFile(path)

	if err != nil {
		return model, err
	}

	err = json.Unmarshal(data, &model)

	return model, err

}

package main

import (
	"flag"
	"fmt"
	"log"

	"classifier/internal/catalog"
	"classifier/internal/classifier"
	"classifier/internal/vocabulary"
)

func main() {
	inputPath := flag.String("input", "../data/commands.csv", "CSV de entrenamiento")
	modelPath := flag.String("model", "../data/vocabulary.json", "archivo JSON del modelo")
	text := flag.String("text", "", "texto o descripción a clasificar")
	train := flag.Bool("train", false, "reentrenar el modelo usando el CSV de entrada")
	minScore := flag.Float64("min-score", classifier.MinScore, "similitud mínima para una predicción concluyente")
	minMargin := flag.Float64("min-margin", classifier.MinMargin, "diferencia mínima entre las dos mejores categorías")
	flag.Parse()

	var model vocabulary.VocabularyModel
	var err error
	if *train {
		commands, err := catalog.ReadCommands(*inputPath)
		if err != nil {
			log.Fatal(err)
		}

		model = vocabulary.BuildVocabulary(commands)
		err = vocabulary.Save(model, *modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Modelo creado: %d categorías en %s\n", len(model.Categories), *modelPath)
	}

	model, err = vocabulary.Load(*modelPath)
	if err != nil {
		log.Fatalf("cargar modelo: %v (ejecutá primero con -train)", err)
	}

	if *text == "" {
		return
	}

	predictions := classifier.PredictTopK(*text, model, 2)
	if len(predictions) == 0 {
		fmt.Println("Sin categoría coincidente")
		return
	}
	if classifier.IsInconclusive(predictions, *minScore, *minMargin) {
		fmt.Println("Resultado inconcluso. Categorías sugeridas:")
		for _, prediction := range predictions {
			fmt.Printf("- %s (%.4f)\n", prediction.Category, prediction.Score)
		}
		return
	}

	fmt.Printf("Categoría: %s\nPuntaje: %.4f\n", predictions[0].Category, predictions[0].Score)

}

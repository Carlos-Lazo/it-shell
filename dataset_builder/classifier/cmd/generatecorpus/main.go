// Command generatecorpus clasifica corpus0.csv y genera un YAML por categoría.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"classifier/internal/classifier"
	"classifier/internal/vocabulary"
)

type document struct {
	Intent  string
	Command string
	Example string
}

func main() {
	inputPath := flag.String("input", "../../webcrawling/debian12/corpus.csv", "CSV del corpus a clasificar")
	modelPath := flag.String("model", "../data/vocabulary.json", "modelo TF-IDF entrenado")
	outputDir := flag.String("output", "../../corpus", "directorio de YAML generados")
	flag.Parse()

	model, err := vocabulary.Load(*modelPath)
	if err != nil {
		log.Fatalf("cargar modelo: %v", err)
	}
	documents, skipped, err := classify(*inputPath, model)
	if err != nil {
		log.Fatal(err)
	}
	if err := writeCorpus(*outputDir, model, documents); err != nil {
		log.Fatal(err)
	}

	count := 0
	for _, rows := range documents {
		count += len(rows)
	}
	fmt.Printf("Documentos clasificados: %d\n", count)
	fmt.Printf("Filas sin coincidencia léxica: %d\n", skipped)
	fmt.Printf("Archivos generados: %d en %s\n", len(model.Categories), *outputDir)
}

// corpus0.csv no tiene encabezados: comando, opción, descripción.
func classify(path string, model vocabulary.VocabularyModel) (map[string][]document, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	grouped := make(map[string][]document, len(model.Categories))
	skipped := 0
	for line := 1; ; line++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, fmt.Errorf("leer fila %d: %w", line, err)
		}
		if len(record) < 3 {
			skipped++
			continue
		}
		command := strings.TrimSpace(record[0])
		option := strings.TrimSpace(record[1])
		intent := strings.TrimSpace(record[2])
		if command == "" || intent == "" {
			skipped++
			continue
		}

		example := command
		if option != "" && option != "SYNOPSIS" {
			example += " " + option
		}
		category, _ := classifier.Predict(example+" "+intent, model)
		if category == "" {
			skipped++
			continue
		}
		grouped[category] = append(grouped[category], document{
			Intent: intent, Command: command, Example: example,
		})
	}
	return grouped, skipped, nil
}

func writeCorpus(outputDir string, model vocabulary.VocabularyModel, documents map[string][]document) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}
	categories := make([]string, 0, len(model.Categories))
	for category := range model.Categories {
		categories = append(categories, category)
	}
	sort.Strings(categories)

	for _, category := range categories {
		path := filepath.Join(outputDir, normalizedCategory(category)+".yaml")
		file, err := os.Create(path)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(file, "category: %s\ndocuments:\n", yamlString(normalizedCategory(category))); err != nil {
			file.Close()
			return err
		}
		for id, document := range documents[category] {
			_, err := fmt.Fprintf(file,
				"- id: %d\n  intent: %s\n  command: %s\n  example: %s\n",
				id+1, yamlString(document.Intent), yamlString(document.Command), yamlString(document.Example))
			if err != nil {
				file.Close()
				return err
			}
		}
		if err := file.Close(); err != nil {
			return err
		}
	}
	return nil
}

// normalizedCategory aplica la convención solicitada para categoría y archivo.
func normalizedCategory(category string) string {
	return strings.ToLower(strings.ReplaceAll(category, "_", ""))
}

// Las cadenas JSON son también escalares YAML válidos y preservan caracteres especiales.
func yamlString(value string) string {
	return strconv.Quote(value)
}

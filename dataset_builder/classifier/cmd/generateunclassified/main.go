// Command generateunclassified writes corpus rows with no lexical match to YAML.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"classifier/internal/classifier"
	"classifier/internal/vocabulary"
)

type document struct {
	intent  string
	command string
	example string
}

func main() {
	inputPath := flag.String("input", "../webcrawling/debian12/corpus0.csv", "CSV del corpus")
	modelPath := flag.String("model", "data/vocabulary.json", "modelo TF-IDF")
	outputPath := flag.String("output", "../corpus/unclassified.yaml", "YAML de salida")
	flag.Parse()

	model, err := vocabulary.Load(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	rows, err := readUnclassified(*inputPath, model)
	if err != nil {
		log.Fatal(err)
	}
	if err := writeYAML(*outputPath, rows); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Documentos sin coincidencia guardados: %d\n", len(rows))
}

func readUnclassified(path string, model vocabulary.VocabularyModel) ([]document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	rows := []document{}
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 3 {
			continue
		}
		command, option, intent := strings.TrimSpace(record[0]), strings.TrimSpace(record[1]), strings.TrimSpace(record[2])
		if command == "" || intent == "" {
			continue
		}
		example := command
		if option != "" && option != "SYNOPSIS" {
			example += " " + option
		}
		category, _ := classifier.Predict(example+" "+intent, model)
		if category == "" {
			rows = append(rows, document{intent: intent, command: command, example: example})
		}
	}
	return rows, nil
}

func writeYAML(path string, rows []document) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := fmt.Fprintln(file, "category: \"unclassified\"\ndocuments:"); err != nil {
		return err
	}
	for id, row := range rows {
		if _, err := fmt.Fprintf(file, "- id: %d\n  intent: %s\n  semantic_text: \"\"\n  command: %s\n  example: %s\n", id+1, quote(row.intent), quote(row.command), quote(row.example)); err != nil {
			return err
		}
	}
	return nil
}

func quote(value string) string { return strconv.Quote(value) }

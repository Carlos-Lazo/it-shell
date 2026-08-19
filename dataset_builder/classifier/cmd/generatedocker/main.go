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
)

type document struct {
	intent  string
	command string
	origen  string
}

func main() {
	inputPath := flag.String("input", "../../../webcrawling/docker/docker_cli_commands_list.csv", "docker CSV del corpus")
	outputPath := flag.String("output", "../../../corpus/docker.yaml", "YAML de salida")
	flag.Parse()

	rows, err := readCSV(*inputPath)
	if err != nil {
		log.Fatal(err)
	}
	if err := writeYAML(*outputPath, rows); err != nil {
		log.Fatal(err)
	}
	root_opath := strings.Split(*outputPath, "/")
	fmt.Printf("Documentos guardados: %d en %s\n", len(rows), root_opath[3]+"/"+root_opath[4])
}

func readCSV(path string) ([]document, error) {
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
		command, intent, url := strings.TrimSpace(record[0]), strings.TrimSpace(record[1]), strings.TrimSpace(record[2])
		if command == "" || intent == "" {
			continue
		}

		rows = append(rows, document{intent: intent, command: command, origen: url})
	}
	return rows, nil
}

func writeYAML(path string, rows []document) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := fmt.Fprintln(file, "category: \"docker\"\ndocuments:"); err != nil {
		return err
	}
	for id, row := range rows {
		if _, err := fmt.Fprintf(file, "- id: %d\n  intent: %s\n  command: %s\n  example: %s\n", id+1, quote(row.intent), quote(row.command), quote(row.origen)); err != nil {
			return err
		}
	}
	return nil
}

func quote(value string) string { return strconv.Quote(value) }

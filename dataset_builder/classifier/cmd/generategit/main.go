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
}

func main() {
	inputPath := flag.String("input", "../../../webcrawling/git/git_commands_list.csv", "git CSV del corpus")
	outputPath := flag.String("output", "../../../corpus/git.yaml", "YAML de salida")
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
	for line := 2; ; line++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 2 {
			continue
		}

		command, intent := strings.TrimSpace(record[0]), strings.TrimSpace(record[1])
		if command == "command" || command == "" || intent == "" {
			continue
		}

		rows = append(rows, document{intent: intent, command: command})
	}
	return rows, nil
}

func writeYAML(path string, rows []document) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := fmt.Fprintln(file, "category: \"git\"\ndocuments:"); err != nil {
		return err
	}

	for id, row := range rows {
		if _, err := fmt.Fprintf(file, "- id: %d\n  intent: %s\n  command: %s\n", id+1, quote(row.intent), quote(row.command)); err != nil {
			return err
		}
	}
	return nil
}

func quote(value string) string { return strconv.Quote(value) }

// Package catalog carga el conjunto de comandos usado para entrenar el modelo.
package catalog

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"classifier/internal/types"
)

// ReadCommands lee un CSV con los encabezados command, description y category.
// Los encabezados se comparan sin distinguir mayúsculas ni espacios.
func ReadCommands(path string) ([]types.Command, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	header, err := reader.Read()
	if err == io.EOF {
		return nil, fmt.Errorf("el archivo CSV está vacío")
	}
	if err != nil {
		return nil, fmt.Errorf("leer encabezados: %w", err)
	}

	indexes, err := headerIndexes(header)
	if err != nil {
		return nil, err
	}

	var commands []types.Command
	for line := 2; ; line++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("leer fila %d: %w", line, err)
		}
		if len(record) <= maxIndex(indexes) {
			return nil, fmt.Errorf("fila %d: faltan columnas", line)
		}

		command := strings.TrimSpace(record[indexes["command"]])
		description := strings.TrimSpace(record[indexes["description"]])
		category := strings.TrimSpace(record[indexes["category"]])
		if command == "" || category == "" {
			return nil, fmt.Errorf("fila %d: command y category son obligatorios", line)
		}
		commands = append(commands, types.Command{Command: command, Description: description, Category: category})
	}

	if len(commands) == 0 {
		return nil, fmt.Errorf("el CSV no contiene comandos")
	}
	return commands, nil
}

func headerIndexes(header []string) (map[string]int, error) {
	indexes := make(map[string]int, len(header))
	for i, value := range header {
		indexes[strings.ToLower(strings.TrimSpace(value))] = i
	}
	for _, required := range []string{"command", "description", "category"} {
		if _, ok := indexes[required]; !ok {
			return nil, fmt.Errorf("falta el encabezado requerido %q", required)
		}
	}
	return indexes, nil
}

func maxIndex(indexes map[string]int) int {
	max := 0
	for _, index := range indexes {
		if index > max {
			max = index
		}
	}
	return max
}

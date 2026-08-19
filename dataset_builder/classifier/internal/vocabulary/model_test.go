package vocabulary

import (
	"testing"

	"classifier/internal/types"
)

func TestBuildVocabularyCreatesOneVectorPerCategory(t *testing.T) {
	commands := []types.Command{
		{Command: "ls", Description: "list files", Category: "files"},
		{Command: "cp", Description: "copy files", Category: "files"},
		{Command: "ps", Description: "show running process", Category: "processes"},
	}

	model := BuildVocabulary(commands)
	if len(model.Categories) != 2 {
		t.Fatalf("categories = %d; want 2", len(model.Categories))
	}
	if model.Categories["files"]["copy"] == 0 {
		t.Fatal("the files vector should contain a TF-IDF weight for copy")
	}
	if model.IDF["files"] == 0 {
		t.Fatal("the model should retain IDF weights for future classifications")
	}
}

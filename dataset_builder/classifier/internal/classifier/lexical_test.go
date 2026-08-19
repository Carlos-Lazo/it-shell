package classifier

import (
	"testing"

	"classifier/internal/types"
	"classifier/internal/vocabulary"
)

func TestPredictSelectsClosestCategory(t *testing.T) {
	model := vocabulary.BuildVocabulary([]types.Command{
		{Command: "ls", Description: "list files directories", Category: "files"},
		{Command: "mkdir", Description: "create directory", Category: "files"},
		{Command: "ps", Description: "display running processes", Category: "processes"},
	})

	category, score := Predict("show active processes", model)
	if category != "processes" || score <= 0 {
		t.Fatalf("Predict() = %q, %v; want processes with a positive score", category, score)
	}
}

func TestIsInconclusiveForLowOrCloseScores(t *testing.T) {
	if !IsInconclusive([]Prediction{{Category: "files", Score: 0.1}}, MinScore, MinMargin) {
		t.Fatal("a low score should be inconclusive")
	}
	if !IsInconclusive([]Prediction{{Category: "files", Score: 0.3}, {Category: "text", Score: 0.28}}, MinScore, MinMargin) {
		t.Fatal("close scores should be inconclusive")
	}
	if IsInconclusive([]Prediction{{Category: "files", Score: 0.3}, {Category: "text", Score: 0.1}}, MinScore, MinMargin) {
		t.Fatal("a high score with a clear margin should be conclusive")
	}
}

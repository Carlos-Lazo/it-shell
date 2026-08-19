package preprocess

import (
	"regexp"
	"strings"
)

var re = regexp.MustCompile(`[^\p{L}\p{N}]+`)

func Tokenize(text string) []string {

	text = strings.ToLower(text)

	text = re.ReplaceAllString(text, " ")

	words := strings.Fields(text)

	return RemoveStopWords(words)
}

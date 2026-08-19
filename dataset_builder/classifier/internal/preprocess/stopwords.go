package preprocess

var stopWords = map[string]struct{}{
	"a":    {},
	"an":   {},
	"the":  {},
	"of":   {},
	"to":   {},
	"and":  {},
	"for":  {},
	"in":   {},
	"on":   {},
	"with": {},
	"as":   {},
	"by":   {},
	"from": {},
	"is":   {},
	"it":   {},
	"or":   {},
	"that": {},
	"this": {},
	"your": {},
}

func RemoveStopWords(words []string) []string {

	out := make([]string, 0, len(words))

	for _, w := range words {

		if _, ok := stopWords[w]; !ok {
			out = append(out, w)
		}

	}

	return out

}

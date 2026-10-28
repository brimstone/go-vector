package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/brimstone/go-embedding"
)

// OpenAIEmbedder is a client for any server exposing the OpenAI embeddings
// API (POST {BaseURL}/embeddings). This covers OpenAI itself as well as
// Ollama (http://localhost:11434/v1), vLLM, llama.cpp's server, LM Studio,
// LocalAI, Text Embeddings Inference and embedding.Handler (see the serve
// command). It implements embedding.Embedder, and therefore also
// vector.Embedder.
type OpenAIEmbedder struct {
	BaseURL string       // e.g. "https://api.openai.com/v1"
	Model   string       // e.g. "text-embedding-3-small"
	APIKey  string       // optional for local servers
	Client  *http.Client // nil = a client with a 60s timeout
}

var _ embedding.Embedder = (*OpenAIEmbedder)(nil)

// Dimension implements embedding.Embedder. It returns 0 because the
// dimension depends on the remote model; callers learn it from the first
// embedding.
func (e *OpenAIEmbedder) Dimension() int { return 0 }

// Embed implements embedding.Embedder.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(map[string]any{"model": e.Model, "input": texts})
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(e.BaseURL, "/") + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	client := e.Client
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("embeddings API: %s: %s", resp.Status, bytes.TrimSpace(msg))
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embeddings API: decoding response: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings API: got %d embeddings for %d inputs", len(out.Data), len(texts))
	}
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data { // the API may return items out of order
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("embeddings API: bad index %d", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
}

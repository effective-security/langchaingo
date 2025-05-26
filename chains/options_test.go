package chains_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
)

func Test_ChainCallOptions(t *testing.T) {
	t.Parallel()
	// Test the default values of ChainCallOptions
	options := &chains.ChainCallOptions{}
	assert.Equal(t, "", options.Model)
	assert.Equal(t, 0, options.MaxTokens)
	assert.Equal(t, 0.0, options.Temperature)
	assert.Empty(t, options.StopWords)
	assert.Nil(t, options.StreamingFunc)
	assert.Equal(t, 0, options.TopK)
	assert.Equal(t, 0.0, options.TopP)
	assert.Equal(t, 0, options.Seed)
	assert.Equal(t, 0, options.MinLength)
	assert.Equal(t, 0, options.MaxLength)
	assert.Empty(t, options.Tools)
	assert.Nil(t, options.ToolChoice)
	assert.Nil(t, options.CallbackHandler)

	llmOpts := chains.GetLLMCallOptions()
	// Only StreamingFunc is set
	assert.Len(t, llmOpts, 1)

	llmOpts = chains.GetLLMCallOptions(
		chains.WithModel("gpt-3.5-turbo"),
		chains.WithMaxTokens(100),
		chains.WithTemperature(0.7),
		chains.WithStopWords([]string{"foo", "bar"}),
		chains.WithTopK(10),
		chains.WithTopP(0.9),
		chains.WithSeed(42),
		chains.WithMinLength(5),
		chains.WithMaxLength(200),
		chains.WithTools([]llms.Tool{
			{
				Type: "tool1",
			},
		}),
		chains.WithToolChoice("tool1"),
		chains.WithCallback(callbacks.StreamLogHandler{}),
	)

	assert.Len(t, llmOpts, 12)
}

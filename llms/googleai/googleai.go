//nolint:all
package googleai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/invopop/jsonschema"
	"github.com/tmc/langchaingo/internal/imageutil"
	"github.com/tmc/langchaingo/llms"
	"google.golang.org/api/iterator"
)

var (
	ErrNoContentInResponse   = errors.New("no content in generation response")
	ErrUnknownPartInResponse = errors.New("unknown part type in generation response")
	ErrInvalidMimeType       = errors.New("invalid mime type on content")
)

const (
	CITATIONS            = "citations"
	SAFETY               = "safety"
	RoleSystem           = "system"
	RoleModel            = "model"
	RoleUser             = "user"
	RoleTool             = "tool"
	ResponseMIMETypeJson = "application/json"
)

// Call implements the [llms.Model] interface.
func (g *GoogleAI) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, g, prompt, options...)
}

// GenerateContent implements the [llms.Model] interface.
func (g *GoogleAI) GenerateContent(
	ctx context.Context,
	messages []llms.MessageContent,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{
		Model:          g.opts.DefaultModel,
		CandidateCount: g.opts.DefaultCandidateCount,
		MaxTokens:      g.opts.DefaultMaxTokens,
		Temperature:    g.opts.DefaultTemperature,
		TopP:           g.opts.DefaultTopP,
		TopK:           g.opts.DefaultTopK,
	}
	for _, opt := range options {
		opt(&opts)
	}

	model := g.client.GenerativeModel(opts.Model)
	model.SetCandidateCount(int32(opts.CandidateCount))
	model.SetMaxOutputTokens(int32(opts.MaxTokens))
	model.SetTemperature(float32(opts.Temperature))
	model.SetTopP(float32(opts.TopP))
	model.SetTopK(int32(opts.TopK))
	model.StopSequences = opts.StopWords
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockThreshold(g.opts.HarmThreshold),
		},
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockThreshold(g.opts.HarmThreshold),
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockThreshold(g.opts.HarmThreshold),
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockThreshold(g.opts.HarmThreshold),
		},
	}
	var err error
	if model.Tools, err = convertTools(opts.Tools); err != nil {
		return nil, err
	}

	// set model.ResponseMIMEType from either opts.JSONMode or opts.ResponseMIMEType
	switch {
	case opts.ResponseMIMEType != "" && opts.JSONMode:
		return nil, fmt.Errorf("conflicting options, can't use JSONMode and ResponseMIMEType together")
	case opts.ResponseMIMEType != "" && !opts.JSONMode:
		model.ResponseMIMEType = opts.ResponseMIMEType
	case opts.ResponseMIMEType == "" && opts.JSONMode && len(model.Tools) == 0:
		model.ResponseMIMEType = ResponseMIMETypeJson
	}

	var response *llms.ContentResponse

	if len(messages) == 1 {
		theMessage := messages[0]
		if theMessage.Role != llms.ChatMessageTypeHuman {
			return nil, fmt.Errorf("got %v message role, want human", theMessage.Role)
		}
		response, err = generateFromSingleMessage(ctx, model, theMessage.Parts, &opts)
	} else {
		response, err = generateFromMessages(ctx, model, messages, &opts)
	}
	if err != nil {
		return nil, err
	}

	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

// convertCandidates converts a sequence of genai.Candidate to a response.
func convertCandidates(candidates []*genai.Candidate, usage *genai.UsageMetadata) (*llms.ContentResponse, error) {
	var contentResponse llms.ContentResponse
	var toolCalls []llms.ToolCall

	for _, candidate := range candidates {
		buf := strings.Builder{}

		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				switch v := part.(type) {
				case genai.Text:
					_, err := buf.WriteString(string(v))
					if err != nil {
						return nil, err
					}
				case genai.FunctionCall:
					b, err := json.Marshal(v.Args)
					if err != nil {
						return nil, err
					}
					toolCall := llms.ToolCall{
						FunctionCall: &llms.FunctionCall{
							Name:      v.Name,
							Arguments: string(b),
						},
					}
					toolCalls = append(toolCalls, toolCall)
				default:
					return nil, ErrUnknownPartInResponse
				}
			}
		}

		metadata := make(map[string]any)
		metadata[CITATIONS] = candidate.CitationMetadata
		metadata[SAFETY] = candidate.SafetyRatings

		if usage != nil {
			metadata["input_tokens"] = usage.PromptTokenCount
			metadata["output_tokens"] = usage.CandidatesTokenCount
			metadata["total_tokens"] = usage.TotalTokenCount
		}

		contentResponse.Choices = append(contentResponse.Choices,
			&llms.ContentChoice{
				Content:        buf.String(),
				StopReason:     candidate.FinishReason.String(),
				GenerationInfo: metadata,
				ToolCalls:      toolCalls,
			})
	}
	return &contentResponse, nil
}

// convertParts converts between a sequence of langchain parts and genai parts.
func convertParts(parts []llms.ContentPart) ([]genai.Part, error) {
	convertedParts := make([]genai.Part, 0, len(parts))
	for _, part := range parts {
		var out genai.Part

		switch p := part.(type) {
		case llms.TextContent:
			out = genai.Text(p.Text)
		case llms.BinaryContent:
			out = genai.Blob{MIMEType: p.MIMEType, Data: p.Data}
		case llms.ImageURLContent:
			typ, data, err := imageutil.DownloadImageData(p.URL)
			if err != nil {
				return nil, err
			}
			out = genai.ImageData(typ, data)
		case llms.ToolCall:
			fc := p.FunctionCall
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(fc.Arguments), &argsMap); err != nil {
				return convertedParts, err
			}
			out = genai.FunctionCall{
				Name: fc.Name,
				Args: argsMap,
			}
		case llms.ToolCallResponse:
			out = genai.FunctionResponse{
				Name: p.Name,
				Response: map[string]any{
					"response": p.Content,
				},
			}
		}

		convertedParts = append(convertedParts, out)
	}
	return convertedParts, nil
}

// convertContent converts between a langchain MessageContent and genai content.
func convertContent(content llms.MessageContent) (*genai.Content, error) {
	parts, err := convertParts(content.Parts)
	if err != nil {
		return nil, err
	}

	c := &genai.Content{
		Parts: parts,
	}

	switch content.Role {
	case llms.ChatMessageTypeSystem:
		c.Role = RoleSystem
	case llms.ChatMessageTypeAI:
		c.Role = RoleModel
	case llms.ChatMessageTypeHuman:
		c.Role = RoleUser
	case llms.ChatMessageTypeGeneric:
		c.Role = RoleUser
	case llms.ChatMessageTypeTool:
		c.Role = RoleUser
	case llms.ChatMessageTypeFunction:
		fallthrough
	default:
		return nil, fmt.Errorf("role %v not supported", content.Role)
	}

	return c, nil
}

// generateFromSingleMessage generates content from the parts of a single
// message.
func generateFromSingleMessage(
	ctx context.Context,
	model *genai.GenerativeModel,
	parts []llms.ContentPart,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	convertedParts, err := convertParts(parts)
	if err != nil {
		return nil, err
	}

	if opts.StreamingFunc == nil {
		// When no streaming is requested, just call GenerateContent and return
		// the complete response with a list of candidates.
		resp, err := model.GenerateContent(ctx, convertedParts...)
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) == 0 {
			return nil, ErrNoContentInResponse
		}
		return convertCandidates(resp.Candidates, resp.UsageMetadata)
	}
	iter := model.GenerateContentStream(ctx, convertedParts...)
	return convertAndStreamFromIterator(ctx, iter, opts)
}

func generateFromMessages(
	ctx context.Context,
	model *genai.GenerativeModel,
	messages []llms.MessageContent,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	history := make([]*genai.Content, 0, len(messages))
	for _, mc := range messages {
		content, err := convertContent(mc)
		if err != nil {
			return nil, err
		}
		if mc.Role == RoleSystem {
			model.SystemInstruction = content
			continue
		}
		history = append(history, content)
	}

	// Given N total messages, genai's chat expects the first N-1 messages as
	// history and the last message as the actual request.
	n := len(history)
	reqContent := history[n-1]
	history = history[:n-1]

	session := model.StartChat()
	session.History = history

	if opts.StreamingFunc == nil {
		resp, err := session.SendMessage(ctx, reqContent.Parts...)
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) == 0 {
			return nil, ErrNoContentInResponse
		}
		return convertCandidates(resp.Candidates, resp.UsageMetadata)
	}
	iter := session.SendMessageStream(ctx, reqContent.Parts...)
	return convertAndStreamFromIterator(ctx, iter, opts)
}

// convertAndStreamFromIterator takes an iterator of GenerateContentResponse
// and produces a llms.ContentResponse reply from it, while streaming the
// resulting text into the opts-provided streaming function.
// Note that this is tricky in the face of multiple
// candidates, so this code assumes only a single candidate for now.
func convertAndStreamFromIterator(
	ctx context.Context,
	iter *genai.GenerateContentResponseIterator,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	candidate := &genai.Candidate{
		Content: &genai.Content{},
	}
DoStream:
	for {
		resp, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			break DoStream
		}
		if err != nil {
			return nil, fmt.Errorf("error in stream mode: %w", err)
		}

		if len(resp.Candidates) != 1 {
			return nil, fmt.Errorf("expect single candidate in stream mode; got %v", len(resp.Candidates))
		}
		respCandidate := resp.Candidates[0]

		if respCandidate.Content == nil {
			break DoStream
		}
		candidate.Content.Parts = append(candidate.Content.Parts, respCandidate.Content.Parts...)
		candidate.Content.Role = respCandidate.Content.Role
		candidate.FinishReason = respCandidate.FinishReason
		candidate.SafetyRatings = respCandidate.SafetyRatings
		candidate.CitationMetadata = respCandidate.CitationMetadata
		candidate.TokenCount += respCandidate.TokenCount

		for _, part := range respCandidate.Content.Parts {
			if text, ok := part.(genai.Text); ok {
				if opts.StreamingFunc(ctx, []byte(text)) != nil {
					break DoStream
				}
			}
		}
	}
	mresp := iter.MergedResponse()
	return convertCandidates([]*genai.Candidate{candidate}, mresp.UsageMetadata)
}

// convertTools converts from a list of langchaingo tools to a list of genai
// tools.
func convertTools(tools []llms.Tool) ([]*genai.Tool, error) {
	genaiTools := make([]*genai.Tool, 0, len(tools))
	for i, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("tool [%d]: unsupported type %q, want 'function'", i, tool.Type)
		}

		// We have a llms.FunctionDefinition in tool.Function, and we have to
		// convert it to genai.FunctionDeclaration
		genaiFuncDecl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		if tool.Function.Parameters != nil {
			var schema *genai.Schema
			var err error

			if jschema, ok := tool.Function.Parameters.(*jsonschema.Schema); ok {
				schema, err = convertJSONSchemaDefinition(jschema)
			} else if params, ok := tool.Function.Parameters.(map[string]any); ok {
				schema, err = convertMapToSchema(params)
			} else {
				return nil, fmt.Errorf("tool [%d]: unsupported type %T of Parameters", i, tool.Function.Parameters)
			}

			if err != nil {
				return nil, fmt.Errorf("tool [%d]: %w", i, err)
			}
			genaiFuncDecl.Parameters = schema
		}

		genaiTools = append(genaiTools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{genaiFuncDecl},
		})
	}

	return genaiTools, nil
}

// convertJSONSchemaDefinition converts a jsonschema.Definition to a genai.Schema.
func convertJSONSchemaDefinition(jschema *jsonschema.Schema) (*genai.Schema, error) {
	schema := &genai.Schema{
		Type:        convertJSONSchemaType(jschema.Type),
		Description: jschema.Description,
		Required:    jschema.Required,
	}

	// Convert properties
	if jschema.Properties != nil {
		schema.Properties = make(map[string]*genai.Schema)
		for pair := jschema.Properties.Oldest(); pair != nil; pair = pair.Next() {
			propSchema, err := convertJSONSchemaDefinition(pair.Value)
			if err != nil {
				return nil, fmt.Errorf("property [%s]: %w", pair.Key, err)
			}
			schema.Properties[pair.Key] = propSchema
		}
	}

	// Convert items for array types
	if jschema.Items != nil {
		itemsSchema, err := convertJSONSchemaDefinition(jschema.Items)
		if err != nil {
			return nil, fmt.Errorf("items: %w", err)
		}
		schema.Items = itemsSchema
	}

	return schema, nil
}

// convertMapToSchema converts a map[string]any to a genai.Schema.
func convertMapToSchema(params map[string]any) (*genai.Schema, error) {
	schema := &genai.Schema{}

	if ty, ok := params["type"]; ok {
		tyString, ok := ty.(string)
		if !ok {
			return nil, fmt.Errorf("expected string for type")
		}
		schema.Type = convertToolSchemaType(tyString)
	}

	if desc, ok := params["description"]; ok {
		descString, ok := desc.(string)
		if !ok {
			return nil, fmt.Errorf("expected string for description")
		}
		schema.Description = descString
	}

	paramProperties, ok := params["properties"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected to find a map of properties")
	}

	schema.Properties = make(map[string]*genai.Schema)
	for propName, propValue := range paramProperties {
		valueMap, ok := propValue.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("property [%v]: expect to find a value map", propName)
		}
		schema.Properties[propName] = &genai.Schema{}

		if ty, ok := valueMap["type"]; ok {
			tyString, ok := ty.(string)
			if !ok {
				return nil, fmt.Errorf("expected string for type")
			}
			schema.Properties[propName].Type = convertToolSchemaType(tyString)
		}
		if desc, ok := valueMap["description"]; ok {
			descString, ok := desc.(string)
			if !ok {
				return nil, fmt.Errorf("expected string for description")
			}
			schema.Properties[propName].Description = descString
		}
	}

	if required, ok := params["required"]; ok {
		if rs, ok := required.([]string); ok {
			schema.Required = rs
		} else if ri, ok := required.([]interface{}); ok {
			rs := make([]string, 0, len(ri))
			for _, r := range ri {
				rString, ok := r.(string)
				if !ok {
					return nil, fmt.Errorf("expected string for required")
				}
				rs = append(rs, rString)
			}
			schema.Required = rs
		} else {
			return nil, fmt.Errorf("expected string for required")
		}
	}

	return schema, nil
}

// convertJSONSchemaType converts a jsonschema.DataType to a genai.Type.
func convertJSONSchemaType(dt string) genai.Type {
	switch dt {
	case "object":
		return genai.TypeObject
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	default:
		return genai.TypeUnspecified
	}
}

// convertToolSchemaType converts a tool's schema type from its langchaingo
// representation (string) to a genai enum.
func convertToolSchemaType(ty string) genai.Type {
	switch ty {
	case "object":
		return genai.TypeObject
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	default:
		return genai.TypeUnspecified
	}
}

// showContent is a debugging helper for genai.Content.
func showContent(w io.Writer, cs []*genai.Content) {
	fmt.Fprintf(w, "Content (len=%v)\n", len(cs))
	for i, c := range cs {
		fmt.Fprintf(w, "[%d]: Role=%s\n", i, c.Role)
		for j, p := range c.Parts {
			fmt.Fprintf(w, "  Parts[%v]: ", j)
			switch pp := p.(type) {
			case genai.Text:
				fmt.Fprintf(w, "Text %q\n", pp)
			case genai.Blob:
				fmt.Fprintf(w, "Blob MIME=%q, size=%d\n", pp.MIMEType, len(pp.Data))
			case genai.FunctionCall:
				fmt.Fprintf(w, "FunctionCall Name=%v, Args=%v\n", pp.Name, pp.Args)
			case genai.FunctionResponse:
				fmt.Fprintf(w, "FunctionResponse Name=%v Response=%v\n", pp.Name, pp.Response)
			default:
				fmt.Fprintf(w, "unknown type %T\n", pp)
			}
		}
	}
}

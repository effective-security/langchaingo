package googleai

import (
	"testing"

	"github.com/google/generative-ai-go/genai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/jsonschema"
	"github.com/tmc/langchaingo/llms"
)

func TestConvertTools(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		tools       []llms.Tool
		expectError bool
		validate    func(t *testing.T, result []*genai.Tool)
	}{
		{
			name: "convert jsonschema.Definition parameters",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "getWeather",
						Description: "Get weather information",
						Parameters: &jsonschema.Definition{
							Type:        jsonschema.Object,
							Description: "Weather request parameters",
							Properties: map[string]jsonschema.Definition{
								"location": {
									Type:        jsonschema.String,
									Description: "City name",
								},
								"unit": {
									Type: jsonschema.String,
									Enum: []string{"celsius", "fahrenheit"},
								},
							},
							Required: []string{"location"},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result []*genai.Tool) {
				require.Len(t, result, 1)
				tool := result[0]
				require.Len(t, tool.FunctionDeclarations, 1)

				funcDecl := tool.FunctionDeclarations[0]
				assert.Equal(t, "getWeather", funcDecl.Name)
				assert.Equal(t, "Get weather information", funcDecl.Description)

				schema := funcDecl.Parameters
				assert.Equal(t, genai.TypeObject, schema.Type)
				assert.Equal(t, "Weather request parameters", schema.Description)
				assert.Equal(t, []string{"location"}, schema.Required)

				// Check properties
				require.Len(t, schema.Properties, 2)

				locationProp := schema.Properties["location"]
				assert.Equal(t, genai.TypeString, locationProp.Type)
				assert.Equal(t, "City name", locationProp.Description)

				unitProp := schema.Properties["unit"]
				assert.Equal(t, genai.TypeString, unitProp.Type)
			},
		},
		{
			name: "convert map[string]any parameters",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "calculateDistance",
						Description: "Calculate distance between two points",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"lat1": map[string]any{
									"type":        "number",
									"description": "Latitude of first point",
								},
								"lon1": map[string]any{
									"type":        "number",
									"description": "Longitude of first point",
								},
								"lat2": map[string]any{
									"type":        "number",
									"description": "Latitude of second point",
								},
								"lon2": map[string]any{
									"type":        "number",
									"description": "Longitude of second point",
								},
							},
							"required": []string{"lat1", "lon1", "lat2", "lon2"},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result []*genai.Tool) {
				require.Len(t, result, 1)
				tool := result[0]
				require.Len(t, tool.FunctionDeclarations, 1)

				funcDecl := tool.FunctionDeclarations[0]
				assert.Equal(t, "calculateDistance", funcDecl.Name)
				assert.Equal(t, "Calculate distance between two points", funcDecl.Description)

				schema := funcDecl.Parameters
				assert.Equal(t, genai.TypeObject, schema.Type)
				assert.Equal(t, []string{"lat1", "lon1", "lat2", "lon2"}, schema.Required)

				// Check properties
				require.Len(t, schema.Properties, 4)

				lat1Prop := schema.Properties["lat1"]
				assert.Equal(t, genai.TypeNumber, lat1Prop.Type)
				assert.Equal(t, "Latitude of first point", lat1Prop.Description)

				lon1Prop := schema.Properties["lon1"]
				assert.Equal(t, genai.TypeNumber, lon1Prop.Type)
				assert.Equal(t, "Longitude of first point", lon1Prop.Description)
			},
		},
		{
			name: "convert map[string]any with interface{} required array",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "searchBooks",
						Description: "Search for books",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"query": map[string]any{
									"type":        "string",
									"description": "Search query",
								},
								"limit": map[string]any{
									"type":        "integer",
									"description": "Maximum number of results",
								},
							},
							"required": []interface{}{"query"},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result []*genai.Tool) {
				require.Len(t, result, 1)
				tool := result[0]
				require.Len(t, tool.FunctionDeclarations, 1)

				funcDecl := tool.FunctionDeclarations[0]
				assert.Equal(t, "searchBooks", funcDecl.Name)

				schema := funcDecl.Parameters
				assert.Equal(t, genai.TypeObject, schema.Type)
				assert.Equal(t, []string{"query"}, schema.Required)

				// Check properties
				require.Len(t, schema.Properties, 2)

				queryProp := schema.Properties["query"]
				assert.Equal(t, genai.TypeString, queryProp.Type)
				assert.Equal(t, "Search query", queryProp.Description)

				limitProp := schema.Properties["limit"]
				assert.Equal(t, genai.TypeInteger, limitProp.Type)
				assert.Equal(t, "Maximum number of results", limitProp.Description)
			},
		},
		{
			name: "convert jsonschema.Definition with array items",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "processItems",
						Description: "Process a list of items",
						Parameters: &jsonschema.Definition{
							Type: jsonschema.Object,
							Properties: map[string]jsonschema.Definition{
								"items": {
									Type: jsonschema.Array,
									Items: &jsonschema.Definition{
										Type:        jsonschema.String,
										Description: "Item to process",
									},
								},
							},
							Required: []string{"items"},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result []*genai.Tool) {
				require.Len(t, result, 1)
				tool := result[0]
				require.Len(t, tool.FunctionDeclarations, 1)

				funcDecl := tool.FunctionDeclarations[0]
				assert.Equal(t, "processItems", funcDecl.Name)

				schema := funcDecl.Parameters
				assert.Equal(t, genai.TypeObject, schema.Type)
				assert.Equal(t, []string{"items"}, schema.Required)

				// Check items property
				require.Len(t, schema.Properties, 1)
				itemsProp := schema.Properties["items"]
				assert.Equal(t, genai.TypeArray, itemsProp.Type)

				// Check items schema
				require.NotNil(t, itemsProp.Items)
				assert.Equal(t, genai.TypeString, itemsProp.Items.Type)
				assert.Equal(t, "Item to process", itemsProp.Items.Description)
			},
		},
		{
			name: "multiple tools",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "tool1",
						Description: "First tool",
						Parameters: &jsonschema.Definition{
							Type: jsonschema.Object,
							Properties: map[string]jsonschema.Definition{
								"param1": {
									Type: jsonschema.String,
								},
							},
						},
					},
				},
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "tool2",
						Description: "Second tool",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"param2": map[string]any{
									"type": "integer",
								},
							},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result []*genai.Tool) {
				require.Len(t, result, 2)

				// Check first tool
				tool1 := result[0]
				require.Len(t, tool1.FunctionDeclarations, 1)
				assert.Equal(t, "tool1", tool1.FunctionDeclarations[0].Name)

				// Check second tool
				tool2 := result[1]
				require.Len(t, tool2.FunctionDeclarations, 1)
				assert.Equal(t, "tool2", tool2.FunctionDeclarations[0].Name)
			},
		},
		{
			name: "unsupported tool type",
			tools: []llms.Tool{
				{
					Type: "unsupported",
					Function: &llms.FunctionDefinition{
						Name: "test",
					},
				},
			},
			expectError: true,
			validate:    func(t *testing.T, result []*genai.Tool) {},
		},
		{
			name: "unsupported parameters type",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "test",
						Description: "Test function",
						Parameters:  "invalid type",
					},
				},
			},
			expectError: true,
			validate:    func(t *testing.T, result []*genai.Tool) {},
		},
		{
			name: "invalid map parameters - missing properties",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "test",
						Description: "Test function",
						Parameters: map[string]any{
							"type": "object",
							// Missing properties
						},
					},
				},
			},
			expectError: true,
			validate:    func(t *testing.T, result []*genai.Tool) {},
		},
		{
			name: "invalid map parameters - invalid type",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "test",
						Description: "Test function",
						Parameters: map[string]any{
							"type":       123, // Should be string
							"properties": map[string]any{},
						},
					},
				},
			},
			expectError: true,
			validate:    func(t *testing.T, result []*genai.Tool) {},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := convertTools(tt.tools)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.validate(t, result)
		})
	}
}

func TestConvertJSONSchemaDefinition(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		definition  *jsonschema.Definition
		expectError bool
		validate    func(t *testing.T, result *genai.Schema)
	}{
		{
			name: "simple object with properties",
			definition: &jsonschema.Definition{
				Type:        jsonschema.Object,
				Description: "Test schema",
				Properties: map[string]jsonschema.Definition{
					"name": {
						Type:        jsonschema.String,
						Description: "Name field",
					},
					"age": {
						Type: jsonschema.Integer,
					},
				},
				Required: []string{"name"},
			},
			expectError: false,
			validate: func(t *testing.T, result *genai.Schema) {
				assert.Equal(t, genai.TypeObject, result.Type)
				assert.Equal(t, "Test schema", result.Description)
				assert.Equal(t, []string{"name"}, result.Required)

				require.Len(t, result.Properties, 2)
				assert.Equal(t, genai.TypeString, result.Properties["name"].Type)
				assert.Equal(t, "Name field", result.Properties["name"].Description)
				assert.Equal(t, genai.TypeInteger, result.Properties["age"].Type)
			},
		},
		{
			name: "array with items",
			definition: &jsonschema.Definition{
				Type: jsonschema.Array,
				Items: &jsonschema.Definition{
					Type:        jsonschema.Number,
					Description: "Array item",
				},
			},
			expectError: false,
			validate: func(t *testing.T, result *genai.Schema) {
				assert.Equal(t, genai.TypeArray, result.Type)
				require.NotNil(t, result.Items)
				assert.Equal(t, genai.TypeNumber, result.Items.Type)
				assert.Equal(t, "Array item", result.Items.Description)
			},
		},
		{
			name: "nested object properties",
			definition: &jsonschema.Definition{
				Type: jsonschema.Object,
				Properties: map[string]jsonschema.Definition{
					"address": {
						Type: jsonschema.Object,
						Properties: map[string]jsonschema.Definition{
							"street": {
								Type: jsonschema.String,
							},
							"city": {
								Type: jsonschema.String,
							},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, result *genai.Schema) {
				assert.Equal(t, genai.TypeObject, result.Type)
				require.Len(t, result.Properties, 1)

				addressProp := result.Properties["address"]
				assert.Equal(t, genai.TypeObject, addressProp.Type)
				require.Len(t, addressProp.Properties, 2)
				assert.Equal(t, genai.TypeString, addressProp.Properties["street"].Type)
				assert.Equal(t, genai.TypeString, addressProp.Properties["city"].Type)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := convertJSONSchemaDefinition(tt.definition)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.validate(t, result)
		})
	}
}

func TestConvertMapToSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		params      map[string]any
		expectError bool
		validate    func(t *testing.T, result *genai.Schema)
	}{
		{
			name: "valid object schema",
			params: map[string]any{
				"type":        "object",
				"description": "Test schema",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "Name field",
					},
					"age": map[string]any{
						"type": "integer",
					},
				},
				"required": []string{"name"},
			},
			expectError: false,
			validate: func(t *testing.T, result *genai.Schema) {
				assert.Equal(t, genai.TypeObject, result.Type)
				assert.Equal(t, "Test schema", result.Description)
				assert.Equal(t, []string{"name"}, result.Required)

				require.Len(t, result.Properties, 2)
				assert.Equal(t, genai.TypeString, result.Properties["name"].Type)
				assert.Equal(t, "Name field", result.Properties["name"].Description)
				assert.Equal(t, genai.TypeInteger, result.Properties["age"].Type)
			},
		},
		{
			name: "missing properties",
			params: map[string]any{
				"type": "object",
			},
			expectError: true,
			validate:    func(t *testing.T, result *genai.Schema) {},
		},
		{
			name: "invalid type value",
			params: map[string]any{
				"type":       123,
				"properties": map[string]any{},
			},
			expectError: true,
			validate:    func(t *testing.T, result *genai.Schema) {},
		},
		{
			name: "invalid description value",
			params: map[string]any{
				"type":        "object",
				"description": 123,
				"properties":  map[string]any{},
			},
			expectError: true,
			validate:    func(t *testing.T, result *genai.Schema) {},
		},
		{
			name: "invalid property type",
			params: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type": 123, // Should be string
					},
				},
			},
			expectError: true,
			validate:    func(t *testing.T, result *genai.Schema) {},
		},
		{
			name: "invalid required array type",
			params: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
				"required":   "not an array",
			},
			expectError: true,
			validate:    func(t *testing.T, result *genai.Schema) {},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := convertMapToSchema(tt.params)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.validate(t, result)
		})
	}
}

func TestConvertJSONSchemaType(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    jsonschema.DataType
		expected genai.Type
	}{
		{jsonschema.Object, genai.TypeObject},
		{jsonschema.String, genai.TypeString},
		{jsonschema.Number, genai.TypeNumber},
		{jsonschema.Integer, genai.TypeInteger},
		{jsonschema.Boolean, genai.TypeBoolean},
		{jsonschema.Array, genai.TypeArray},
		{jsonschema.Null, genai.TypeUnspecified},
		{"unknown", genai.TypeUnspecified},
	}

	for _, tt := range tests {
		t.Run(string(tt.input), func(t *testing.T) {
			t.Parallel()
			result := convertJSONSchemaType(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

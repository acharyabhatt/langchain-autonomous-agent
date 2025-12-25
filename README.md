# 🤖 LangChain AI Agent with Custom Tools

An intelligent AI agent built with LangChain that can use multiple tools to accomplish complex tasks autonomously.

## 🎯 Features

- **8 Built-in Tools**: Calculator, Wikipedia, Web Search, Weather, Web Scraper, Python REPL, File Reader/Writer
- **ReAct Framework**: Reasoning and acting in an interleaved manner
- **Conversation Memory**: Maintains context across interactions
- **Extensible**: Easy to add custom tools
- **Local LLM Support**: Works with Ollama (no API keys required)
- **Interactive CLI**: Chat interface for easy interaction

## 🛠️ Available Tools

1. **Calculator** - Mathematical calculations
2. **Wikipedia** - Factual information lookup
3. **WebSearch** - Internet search via DuckDuckGo
4. **Weather** - Current weather information
5. **WebScraper** - Extract text from web pages
6. **PythonREPL** - Execute Python code
7. **FileWriter** - Write to text files
8. **FileReader** - Read from text files

## 📦 Installation

### Prerequisites

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull a model:
```bash
ollama pull llama2
# or
ollama pull mistral
```

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd langchain-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Interactive Mode

```bash
python agent.py
```

Example interactions:

```
You: What's the weather in London?
Agent: [Uses Weather tool to fetch current weather]

You: Calculate the square root of 144
Agent: [Uses Calculator tool] Result: 12.0

You: Search for recent AI news
Agent: [Uses WebSearch tool to find latest AI news]

You: Write a summary to notes.txt|AI is advancing rapidly
Agent: [Uses FileWriter tool] Successfully wrote to notes.txt
```

### Programmatic Usage

```python
from agent import LangChainAgent

# Initialize agent
agent = LangChainAgent(model_name="llama2", temperature=0.7)

# Run queries
response = agent.run("What's 25 times 4?")
print(response)

# Get available tools
tools = agent.get_tool_descriptions()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
```

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangChain      │
│  ReAct Agent    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│         Tool Selection          │
│  (Based on query requirements)  │
└────────┬────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│              Execute Tool              │
│  ┌──────────┐  ┌──────────┐          │
│  │Calculator│  │Wikipedia │  ...     │
│  └──────────┘  └──────────┘          │
└────────┬───────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

## 🔧 Adding Custom Tools

Create a new tool function:

```python
@staticmethod
def my_custom_tool(input_text: str) -> str:
    """
    Description of what the tool does.
    Input format explanation.
    """
    try:
        # Tool logic here
        result = process(input_text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

Add it to the tools list in `_create_tools()`:

```python
Tool(
    name="MyCustomTool",
    func=custom_tools.my_custom_tool,
    description="What this tool does and when to use it."
)
```

## 📊 Example Use Cases

### Research Assistant
```python
query = """
Find information about quantum computing on Wikipedia,
then search for the latest quantum computing news,
and write a summary to quantum_summary.txt
"""
agent.run(query)
```

### Data Analysis
```python
query = """
Calculate the mean of [10, 20, 30, 40, 50],
then write the result to stats.txt
"""
agent.run(query)
```

### Web Research
```python
query = """
Search for information about machine learning trends,
scrape the top result, and summarize the key points
"""
agent.run(query)
```

## ⚙️ Configuration

Modify agent parameters in initialization:

```python
agent = LangChainAgent(
    model_name="mistral",      # LLM model
    temperature=0.7            # Creativity level (0-1)
)
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test individual tools
python -c "from agent import CustomTools; print(CustomTools.calculator('2+2'))"
```

## 📈 Performance Tips

1. **Model Selection**: 
   - Llama2: Best for complex reasoning
   - Mistral: Faster, good for simple tasks
   
2. **Temperature**: 
   - 0.0-0.3: Deterministic, factual
   - 0.4-0.7: Balanced
   - 0.8-1.0: Creative, exploratory

3. **Max Iterations**: Increase for complex multi-step tasks

## 🔒 Security Considerations

- **PythonREPL**: Can execute arbitrary code - use with caution
- **FileWriter**: Can overwrite files - validate inputs
- **WebScraper**: Respect robots.txt and rate limits
- **API Keys**: Store in `.env` file, never commit

## 🐛 Troubleshooting

**Agent not responding:**
- Check if Ollama is running: `ollama list`
- Verify model is downloaded: `ollama pull llama2`

**Tool errors:**
- Check internet connection for Wikipedia/Search
- Verify file permissions for FileReader/Writer
- Ensure valid Python syntax for PythonREPL

**Memory issues:**
- Clear conversation history periodically
- Use smaller context models
- Reduce max_iterations

## 🚀 Future Enhancements

- [ ] Add database query tool
- [ ] Implement email sending
- [ ] Add image generation capability
- [ ] Create GUI interface
- [ ] Add voice input/output
- [ ] Implement multi-agent collaboration

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new tools
4. Submit a pull request

## 📝 License

MIT License

## 🙏 Acknowledgments

- LangChain for the framework
- Ollama for local LLM support
- All tool API providers

## 📧 Contact

For questions or suggestions, open an issue on GitHub.

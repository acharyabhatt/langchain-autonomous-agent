"""
LangChain AI Agent with Custom Tools
An intelligent agent that can use multiple tools to accomplish tasks
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.utilities import WikipediaAPIWrapper, PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
from bs4 import BeautifulSoup
import json
import math


class CustomTools:
    """Collection of custom tools for the agent"""
    
    @staticmethod
    def calculator(expression: str) -> str:
        """
        Performs mathematical calculations.
        Input should be a valid Python mathematical expression.
        Example: "2 + 2" or "math.sqrt(16)"
        """
        try:
            # Safe evaluation of mathematical expressions
            result = eval(expression, {"__builtins__": {}, "math": math}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    @staticmethod
    def weather(location: str) -> str:
        """
        Gets current weather for a location.
        Input should be a city name or location.
        Example: "London" or "New York"
        """
        try:
            # Using wttr.in for weather (no API key required)
            url = f"https://wttr.in/{location}?format=j1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data['current_condition'][0]
                
                weather_info = f"""
Weather for {location}:
- Temperature: {current['temp_C']}°C / {current['temp_F']}°F
- Condition: {current['weatherDesc'][0]['value']}
- Humidity: {current['humidity']}%
- Wind Speed: {current['windspeedKmph']} km/h
                """
                return weather_info.strip()
            else:
                return f"Could not fetch weather for {location}"
                
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    @staticmethod
    def web_scraper(url: str) -> str:
        """
        Extracts text content from a webpage.
        Input should be a valid URL.
        Example: "https://example.com"
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit to first 500 characters
                return text[:500] + "..." if len(text) > 500 else text
            else:
                return f"Failed to fetch URL: Status code {response.status_code}"
                
        except Exception as e:
            return f"Error scraping web page: {str(e)}"
    
    @staticmethod
    def file_writer(content: str) -> str:
        """
        Writes content to a text file.
        Input should be in format: "filename.txt|content to write"
        Example: "notes.txt|This is my note"
        """
        try:
            filename, text = content.split("|", 1)
            
            with open(filename, 'w') as f:
                f.write(text)
            
            return f"Successfully wrote to {filename}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    @staticmethod
    def file_reader(filename: str) -> str:
        """
        Reads content from a text file.
        Input should be a filename.
        Example: "notes.txt"
        """
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            return content if content else "File is empty"
            
        except FileNotFoundError:
            return f"File '{filename}' not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class LangChainAgent:
    """LangChain Agent with multiple tools"""
    
    def __init__(self, model_name: str = "llama2", temperature: float = 0.7):
        """Initialize the agent with tools"""
        
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = Ollama(
            model=self.model_name,
            temperature=self.temperature,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
    def _create_tools(self) -> List[Tool]:
        """Create and return list of tools"""
        
        # Initialize external tools
        wikipedia = WikipediaAPIWrapper()
        search = DuckDuckGoSearchRun()
        python_repl = PythonREPL()
        
        # Create custom tools
        custom_tools = CustomTools()
        
        tools = [
            Tool(
                name="Calculator",
                func=custom_tools.calculator,
                description="Useful for mathematical calculations. Input should be a valid Python expression."
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for looking up factual information on Wikipedia. Input should be a search query."
            ),
            Tool(
                name="WebSearch",
                func=search.run,
                description="Useful for searching the internet for current information. Input should be a search query."
            ),
            Tool(
                name="Weather",
                func=custom_tools.weather,
                description="Gets current weather for a location. Input should be a city name."
            ),
            Tool(
                name="WebScraper",
                func=custom_tools.web_scraper,
                description="Extracts text from a webpage. Input should be a URL."
            ),
            Tool(
                name="PythonREPL",
                func=python_repl.run,
                description="Executes Python code. Input should be valid Python code. Use for complex computations."
            ),
            Tool(
                name="FileWriter",
                func=custom_tools.file_writer,
                description="Writes content to a file. Input format: 'filename.txt|content'"
            ),
            Tool(
                name="FileReader",
                func=custom_tools.file_reader,
                description="Reads content from a file. Input should be a filename."
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        
        # Create prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def run(self, query: str) -> str:
        """Run the agent with a query"""
        try:
            result = self.agent.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of available tools"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]


def main():
    """Main execution function"""
    print("=" * 80)
    print("LangChain AI Agent with Tools")
    print("=" * 80)
    print("\nInitializing agent...")
    
    # Initialize agent
    agent = LangChainAgent(model_name="llama2", temperature=0.7)
    
    print("\nAvailable Tools:")
    for tool in agent.get_tool_descriptions():
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n" + "=" * 80)
    print("Agent ready! Type 'quit' to exit.\n")
    
    # Interactive loop
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nAgent: ")
        response = agent.run(query)
        print(f"\n{response}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()

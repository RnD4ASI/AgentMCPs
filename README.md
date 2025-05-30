# MCP-Inspired AI Agent Library

## 1. Overview

This library provides a collection of AI agents designed to interact with various functionalities (termed "skills") through a common framework inspired by the Model Context Protocol (MCP). Its primary purpose is to demonstrate how different Large Language Models (LLMs) from various providers can be integrated into a shared operational structure. This structure emphasizes pluggable skills, persistent episodic memory (factual and procedural), and a standardized way to define and invoke capabilities.

The project showcases a modular approach to building AI agent systems, where the core intelligence (LLM) can be swapped, and new functionalities can be added as discrete skills without altering the agent's fundamental operational logic.

## 2. Model Context Protocol (MCP) Alignment

This library implements a practical interpretation of ideas from the Model Context Protocol. It utilizes a central JSON configuration file, `mcp_configurations.json`, which acts as a registry for defining:

*   **Models**: Available LLMs that can be used by the agents.
*   **Skills**: Tools or functionalities that agents can invoke. Each skill has defined parameters, expected context, and a designated handler.
*   **Context Types**: Standardized data structures for information that can be passed to skills (e.g., memory snapshots, user queries).

While not an official or complete implementation of the MCP specification, this library aims to apply its core principles of modularity, standardized interaction patterns, and explicit context management in building AI agent systems. For official information on the Model Context Protocol, please refer to [modelcontextprotocol.io](https://modelcontextprotocol.io).

## 3. Library Components

### `mcp_configurations.json`

This crucial file serves as the central schema and registry for the entire system. It defines:
*   **`mcp_version`**: The version of the MCP-like schema being used.
*   **`models`**: An array of LLM definitions, specifying their ID, provider, API details (including environment variables for keys), and capabilities.
*   **`skills`**: An array of skill definitions, including:
    *   `id`: Unique skill identifier.
    *   `name` and `description`.
    *   `handler_module` and `handler_class_or_function`: Specifies the Python module and the class/method or function that implements the skill's logic.
    *   `parameters`: An array defining the input parameters the skill expects (name, type, description, required).
    *   `expected_context`: A list of context type IDs that the skill might utilize.
    *   `output_schema`: A JSON schema for the skill's output.
*   **`context_types`**: An array defining standard data structures for context information, each with an `id`, `description`, and a JSON `schema`.

### Skills

Skill implementations are located in separate Python files:

*   **`realtime_processing_skill.py`**:
    *   **Purpose**: Provides functionality for real-time data validation and transformation. It can process a stream of data items, validate them against rules, and apply transformations.
    *   **Handler**: `RealTimeStreamingMCP.process_stream_data`
*   **`federated_learning_skill.py`**:
    *   **Purpose**: Simulates federated learning operations like starting new rounds, accepting model updates from participants, and aggregating these updates to a global model.
    *   **Handler**: `FederatedLearningMCP.execute_action` (actions specified via parameters)
*   **`robust_messaging_skill.py`**:
    *   **Purpose**: Implements robust message processing with features like automatic retries (with exponential backoff) for transient failures and a simulated dead-letter queue (DLQ) for messages that fail persistently.
    *   **Handler**: `RobustMCP.process_message`

### AI Agents

Each agent provides an interface to a specific LLM backend but adheres to the common MCP-based skill interaction pattern. They are responsible for understanding user queries (using their respective LLMs), selecting an appropriate skill from the `mcp_configurations.json` registry, preparing necessary context, invoking the skill, and processing its response.

*   `azure_openai_agent.py`: Agent using Azure OpenAI services.
*   `openai_agent.py`: Agent using OpenAI API.
*   `anthropic_agent.py`: Agent using Anthropic Claude API.
*   `gemini_agent.py`: Agent using Google Gemini API.
*   `huggingface_agent.py`: Agent using a local Hugging Face Transformer model.

### Episodic Memory

Each agent maintains its own persistent episodic memory, crucial for providing context to skills and maintaining conversation history:

*   **Factual Memory** (`*_factual_memory.json`): Stores key-value pairs representing facts learned or preferences stated by the user (e.g., "user_location": "Paris"). This corresponds to the `factual_memory_snapshot` context type.
*   **Procedural Memory** (`*_procedural_memory.json`): Records tasks performed or sequences of actions taken by the agent (e.g., "skill_invoked: robust_message_handler"). This corresponds to the `procedural_memory_snapshot` context type.

This memory is loaded at agent initialization and saved upon modification.

## 4. Setup

### Requirements

*   Python 3.8+
*   Necessary Python libraries are listed in `requirements.txt`.

### Installation

1.  Clone the repository.
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### API Keys

To use the agents that connect to commercial LLM services, you need to set the appropriate API keys as environment variables. The specific environment variable names expected by each model provider are defined in `mcp_configurations.json` under the `models -> api_details -> *_env_var` fields.

Common examples include:
*   For Azure OpenAI: `AZURE_OAI_KEY`, `AZURE_OAI_ENDPOINT`, `AZURE_OAI_LLM_DEPLOYMENT`
*   For OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL_NAME` (optional, defaults in agent)
*   For Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL_NAME` (optional, defaults in agent)
*   For Google Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL_NAME` (optional, defaults in agent)
*   For HuggingFace (local model device selection): `HF_DEVICE` (e.g., `-1` for CPU, `0` for GPU), `HF_MODEL_NAME` (optional, defaults in agent).

Please refer to `mcp_configurations.json` for the exact environment variable names associated with each model definition.

## 5. Basic Usage

The following snippet demonstrates how to import and use an agent (e.g., `OpenAIAgent`):

```python
import os
from openai_agent import OpenAIAgent # Or any other agent

# Ensure your API key is set as an environment variable
# For example, for OpenAIAgent:
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
# (Better to set this in your shell environment)

if __name__ == '__main__':
    # Check if mcp_configurations.json exists
    if not os.path.exists("mcp_configurations.json"):
        print("Error: mcp_configurations.json not found. Please ensure the file is in the root directory.")
        exit()

    # Check if skill files exist (basic check for one)
    if not os.path.exists("realtime_processing_skill.py"):
        print("Error: Skill files (e.g., realtime_processing_skill.py) not found. Ensure they are in the root directory.")
        exit()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set. Skipping OpenAI agent demo.")
    else:
        try:
            # Instantiate the agent
            agent = OpenAIAgent(api_key=api_key)

            # Make a sample query
            query1 = "What is the capital of France and remember I prefer very short answers."
            print(f"\nUser Query: {query1}")
            response1 = agent.process_query(query1)
            print(f"Agent Response: {response1}")

            # Example query that might use the robust messaging skill
            query2 = "Please handle this message with robust delivery: '{\"id\":\"msg123\", \"payload\":{\"data\":\"important info\"}}'"
            print(f"\nUser Query: {query2}")
            response2 = agent.process_query(query2)
            print(f"Agent Response: {response2}")

        except RuntimeError as e:
            print(f"Runtime Error during agent usage: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

```

When `agent.process_query()` is called:
1.  The agent uses its configured LLM to analyze the query and `mcp_configurations.json`.
2.  The LLM determines the user's intent, identifies an appropriate `skill_id` (if any), and extracts necessary parameters for that skill.
3.  The agent prepares the required `context_data` for the skill, potentially including snapshots of its factual and procedural memory.
4.  The agent dynamically loads and invokes the identified skill, passing the parameters and context.
5.  The skill executes its logic and returns a standardized dictionary response.
6.  The agent processes this response (handling success or errors) and formulates a final reply for the user.
7.  The agent may also update its internal memory based on the interaction.

## 6. Running Unit Tests

The library includes a suite of unit tests to verify the functionality of individual components (MCP configurations, skills, and agents).

To run the tests, navigate to the project root directory in your terminal and execute:

```bash
python -m unittest discover tests
```

Ensure all dependencies (including test-related ones if any, though `unittest` is standard) are installed and any necessary environment variables (like API keys for mocked tests that might still initialize clients) are either set or the tests correctly mock client instantiation.

## 7. Project Structure

```
.
├── mcp_configurations.json       # Central registry for models, skills, context types
├── realtime_processing_skill.py  # Skill: Real-time data processing logic
├── federated_learning_skill.py   # Skill: Federated learning simulation logic
├── robust_messaging_skill.py     # Skill: Robust message handling logic
├── azure_openai_agent.py         # Agent: Azure OpenAI backend
├── openai_agent.py               # Agent: OpenAI backend
├── anthropic_agent.py            # Agent: Anthropic Claude backend
├── gemini_agent.py               # Agent: Google Gemini backend
├── huggingface_agent.py          # Agent: Local HuggingFace model backend
├── azure_agent_factual_memory.json   # Example memory file (created on run)
├── openai_agent_procedural_memory.json # Example memory file (created on run)
├── ...                           # Other agent memory files
├── requirements.txt              # Python package dependencies
├── README.md                     # This file
└── tests/                        # Directory for unit tests
    ├── test_mcp_configurations.py
    ├── test_mcp_skills.py
    ├── test_azure_openai_agent.py
    ├── test_openai_agent.py
    ├── test_anthropic_agent.py
    ├── test_gemini_agent.py
    └── test_huggingface_agent.py
```

This structure promotes modularity and clear separation of concerns between the agent's core logic, the skills they can use, and the LLM backends.
```

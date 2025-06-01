# User Guide

This guide provides instructions on how to set up the environment, run the individual AI agents, execute the A2A collaboration demo, run unit tests, and offers brief notes on extending the system.

## 1. Environment Setup

### Prerequisites
*   **Python**: Ensure you have Python installed (version 3.8 or higher is recommended).
*   **pip**: Python's package installer, usually comes with Python.

### Steps

1.  **Clone the Repository**:
    If you haven't already, clone the repository to your local machine.
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended)**:
    It's good practice to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    This will install `openai`, `anthropic`, `requests` (for the A2A client, though calls are currently mocked), and the `a2a-sdk` (though the project currently uses custom A2A classes).

4.  **Set Environment Variables (API Keys)**:
    To run the agents that interact with external services (OpenAI, Anthropic, Gemini, Azure OpenAI), you need to set their respective API keys as environment variables.

    *   **For OpenAI Agent**:
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
        ```
    *   **For Anthropic Agent**:
        ```bash
        export ANTHROPIC_API_KEY="your_anthropic_api_key"
        ```
    *   **For Gemini Agent**:
        ```bash
        export GEMINI_API_KEY="your_gemini_api_key"
        ```
        (Note: The Gemini agent (`gemini_agent.py`) in the provided files uses a placeholder `GoogleGenerativeAI` class. You'd need to replace this with actual Gemini SDK usage and corresponding API key setup if you were to fully implement it.)
    *   **For Azure OpenAI Agent**:
        You'll need to set several variables, typically:
        ```bash
        export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
        export AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
        export AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
        export OPENAI_API_VERSION="your_api_version"
        ```
        (The `azure_openai_agent.py` expects these, ensure they match its `__init__`.)

    You can set these in your terminal session or add them to your shell's configuration file (e.g., `.bashrc`, `.zshrc`, or use a `.env` file with a library like `python-dotenv` if you prefer, though that's not set up by default).

## 2. Running Individual Agents

Each agent script (`openai_agent.py`, `anthropic_agent.py`, `gemini_agent.py`, `azure_openai_agent.py`, `huggingface_agent.py`) has an `if __name__ == '__main__':` block that demonstrates its basic functionality. This usually involves initializing the agent and processing a few sample queries.

**To run an agent (e.g., OpenAI Agent):**

1.  Ensure your virtual environment is activated and API keys are set.
2.  Navigate to the root directory of the project.
3.  Execute the script:
    ```bash
    python openai_agent.py
    ```
    The script will print output to the console, showing the agent's responses to predefined queries and its generated Agent Card for A2A.

    Similarly, for other agents:
    ```bash
    python anthropic_agent.py
    python gemini_agent.py
    # (and so on for azure_openai_agent.py, huggingface_agent.py)
    ```

**Expected Output for Individual Agents:**
*   Initialization logs.
*   Responses to sample queries defined in their `main` block.
*   A JSON representation of their A2A Agent Card.
*   Final memory states (factual and procedural).

## 3. Running the A2A Collaboration Demo

The `a2a_collaboration_demo.py` script demonstrates a simulated interaction between the `OpenAIAgent` and `AnthropicAgent` using A2A protocol concepts.

**To run the demo:**

1.  Ensure your virtual environment is activated.
2.  **Crucially, set both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables.**
3.  Navigate to the root directory.
4.  Execute the script:
    ```bash
    python a2a_collaboration_demo.py
    ```

**Expected Output for Collaboration Demo:**
*   Logs indicating initialization of both OpenAI and Anthropic agents.
*   JSON representations of both agents' A2A Agent Cards.
*   Logs detailing the task creation by the OpenAI agent for the Anthropic agent.
*   Logs showing the **simulated** A2A call (the demo calls the Anthropic agent's `handle_a2a_task_request` method directly).
*   The (mocked) response from the Anthropic agent.
*   Information about the task status and any artifacts produced.

## 4. Running Unit Tests

Unit tests are provided in the `tests/test_a2a_integration.py` file. These tests cover A2A protocol components, agent A2A compliance features, and a smoke test for the collaboration demo.

**To run the tests:**

1.  Ensure your virtual environment is activated.
2.  Navigate to the root directory of the project.
3.  Run the tests using Python's `unittest` module:
    ```bash
    python -m unittest tests.test_a2a_integration
    ```
    Or, more simply:
    ```bash
    python -m unittest
    ```
    (This command will discover tests in the `tests` directory if it's structured as a package with an `__init__.py` file).

**Expected Output for Unit Tests:**
*   Dots (`.`) for passing tests, `F` for failures, `E` for errors.
*   A summary of the number of tests run and their outcomes (e.g., `OK` if all pass).
*   The tests are designed to use mock API keys, so you don't need real keys set for the tests themselves.

## 5. Extending the System

### Adding a New Agent

1.  Create a new Python file for your agent (e.g., `my_new_agent.py`).
2.  Define your agent class, likely following the structure of existing agents (e.g., `__init__`, `process_query`, memory management, MCP loading).
3.  **A2A Compliance**:
    *   Import `AgentCard`, `A2ATask`, `A2AClient` from `a2a_protocol.py`.
    *   In `__init__`: initialize `self.a2a_client`, `self.agent_id`, `self.a2a_endpoint_url`.
    *   Implement `get_agent_card()`: Define its capabilities.
    *   Implement `handle_a2a_task_request()`: Logic to process tasks sent to this agent.
    *   Add `# TODO: A2A Delegation Point` comments in its query processing logic if it might delegate tasks.
4.  Add an `if __name__ == '__main__':` block for standalone testing.
5.  Consider adding it to the `a2a_collaboration_demo.py` or creating new demos.
6.  Add unit tests for its A2A features.

### Adding a New Custom MCP

1.  Create a new Python file for your MCP (e.g., `custom_mcp_4.py`).
2.  Define your MCP class with methods providing specific functionalities. Include docstrings.
3.  Update `CUSTOM_MCP_MODULES` dictionary in the agent files where this MCP should be available, providing its module name, class name, and a configuration key if needed.
    ```python
    CUSTOM_MCP_MODULES = {
        # ... existing MCPs ...
        "MyNewMCP": {
            "module_name": "custom_mcp_4", # Your new MCP file name (without .py)
            "class_name": "MyNewMCP",      # Your new MCP class name
            "config_key": "my_new_mcp_config" # Optional: if it needs specific config
        }
    }
    ```
4.  If the new MCP is intended to be an A2A capability, ensure the agent's `get_agent_card()` method reflects this new capability in its list.
5.  The agent's `_determine_intent_and_params_with_llm` prompt might need adjustment if you want the LLM to specifically recommend this new MCP tool.

This guide should help you get started with using and understanding the codebase. For more details on the A2A and MCP protocols themselves, refer to the `MCP_A2A_INTEGRATION.md` document.
```

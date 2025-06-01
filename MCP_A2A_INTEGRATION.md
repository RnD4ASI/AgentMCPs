# Integrating MCP (Model Context Protocol) and A2A (Agent-to-Agent) Protocol

This document explains how the Model Context Protocol (MCP) and the Agent-to-Agent (A2A) protocol are used together in this project, how to configure A2A for the agents, and provides an overview of the demonstration script.

## MCP and A2A: Complementary Protocols

**Model Context Protocol (MCP):**
- MCP is designed to provide **internal context and capabilities** to an individual AI agent.
- It allows an agent to understand its available tools, data sources, and operational guidelines.
- In this project, MCP is implemented through:
    - `mcp_configurations.json`: Defines shared protocol configurations that agents can use.
    - Custom MCP Modules (`custom_mcp_1.py`, `custom_mcp_2.py`, `custom_mcp_3.py`): These are specialized tools (e.g., for streaming, federated learning, robust messaging) that an agent can leverage. An agent's `__init__` method loads these MCPs.
    - The LLM call (`_determine_intent_and_params_with_llm`): Used by agents to understand user queries and map them to available MCPs or general knowledge.

**Agent-to-Agent Protocol (A2A):**
- A2A is designed for **external communication and collaboration between different AI agents**.
- It provides a standardized way for agents to discover each other, delegate tasks, and exchange information, even if they are built with different underlying technologies or by different vendors.
- Key A2A concepts used in this project (from `a2a_protocol.py`):
    - **Agent Card**: A JSON description of an agent, detailing its ID, name, capabilities, and A2A communication endpoint. Each agent in this project can generate its Agent Card via the `get_agent_card()` method. Capabilities in the Agent Card are typically derived from the MCPs the agent can use or its core functionalities.
    - **A2A Task**: A structured request sent from a client agent to a remote agent, specifying the capability to be invoked and the necessary parameters.
    - **A2A Client/Server**: Agents can act as A2A clients (sending tasks using `A2AClient`) and A2A servers (handling incoming tasks via `handle_a2a_task_request`).

**How MCP and A2A Work Together:**

1.  **Internal Capability Definition (MCP)**: An agent first understands its own capabilities through its MCP setup (loaded tools, configurations).
2.  **Decision Making (MCP + Agent Logic)**: When an agent processes a query (e.g., via its `process_query` method), it uses its MCP-enhanced LLM to understand the intent. Based on this, the agent decides:
    *   Can it handle the request using its own MCP tools and knowledge?
    *   Is the request better suited for another agent with specialized skills?
3.  **External Collaboration (A2A)**: If the agent decides to delegate, it uses A2A:
    *   **Discovery (Conceptual)**: It would first need to discover the target agent and its capabilities (e.g., by looking up an Agent Card in a directory, though this demo uses pre-configured knowledge).
    *   **Task Invocation (A2A Client)**: It constructs an `A2ATask` and uses its `a2a_client` to send the request to the remote agent's A2A endpoint.
4.  **Task Execution (Remote Agent - MCP + A2A Server)**: The remote agent receives the `A2ATask` via its A2A endpoint (handled by `handle_a2a_task_request`). It then uses its own MCP tools and internal logic to execute the requested capability and returns the result as an A2A response.

In essence, MCP equips an agent internally, while A2A enables it to interact with the external agent ecosystem.

## Configuring A2A for Agents

A2A configuration primarily involves defining how agents present themselves (Agent Cards) and where they listen for A2A communication.

1.  **Agent ID and Endpoint URL**:
    *   Each agent instance has a unique `self.agent_id` (e.g., `openai-agent-<uuid>`).
    *   Each agent has a designated `self.a2a_endpoint_url` (e.g., `http://localhost:8001/a2a`). This is the URL where the agent would listen for incoming A2A requests if it were running an HTTP server. (Note: The actual HTTP server implementation is not part of the current agent classes but is anticipated by this setup).

2.  **Agent Card Generation (`get_agent_card()` method)**:
    *   Each agent class (`OpenAIAgent`, `AnthropicAgent`, etc.) has a `get_agent_card()` method.
    *   This method returns an `AgentCard` object, populating fields like:
        *   `agentId`: The agent's unique ID.
        *   `name`: A human-readable name (e.g., "OpenAI Assistant Agent").
        *   `description`: A brief description of the agent.
        *   `version`: The agent's software version.
        *   `protocolVersion`: The A2A protocol version being used.
        *   `capabilities`: A list of actions the agent can perform. These are derived from its `CUSTOM_MCP_MODULES` or its core functions. Each capability includes:
            *   `name`: A unique name for the capability (e.g., "RealTimeStreamingMCP_process_stream_data").
            *   `description`: What the capability does.
            *   `parameters`: Input parameters required by the capability (name, type, required, description).
            *   `returns`: Description of the output.
        *   `endpointUrl`: The agent's A2A listening URL.

3.  **A2A Client (`self.a2a_client`)**:
    *   Initialized in each agent's `__init__`.
    *   Used to send tasks to other agents. Currently, its `send_task_request` is a placeholder in `a2a_protocol.py` that logs and returns a mock response, but it's designed to make HTTP POST requests in a full implementation.

4.  **A2A Task Handling (`handle_a2a_task_request(self, task_data: dict)`)**:
    *   A placeholder method in each agent designed to be called when an A2A task request is received (e.g., by an HTTP server routing to this method).
    *   It parses the incoming `task_data` into an `A2ATask` object.
    *   It currently simulates task execution based on `task.capability_name` and returns a mock JSON-RPC response.

**Environment Variables:**
*   While not strictly A2A configuration, agents require their respective API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) set as environment variables to initialize correctly. These are used by the agents' core functionalities, which in turn might be exposed via A2A capabilities.

## Agent Collaboration Demonstration (`a2a_collaboration_demo.py`)

The `a2a_collaboration_demo.py` script showcases a basic interaction between an `OpenAIAgent` and an `AnthropicAgent` using the A2A protocol concepts.

**Overview:**

1.  **Initialization**:
    *   Sets up logging.
    *   Loads `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from environment variables.
    *   Instantiates `OpenAIAgent` and `AnthropicAgent`.

2.  **Agent Card Display**:
    *   Retrieves and prints the JSON representation of each agent's `AgentCard` using their `get_agent_card()` method.

3.  **Task Definition**:
    *   A scenario is created where the `OpenAIAgent` (client) wants to delegate a task to the `AnthropicAgent` (remote).
    *   It identifies a capability from the `AnthropicAgent`'s card (e.g., "RobustMCP_process_message" or the first available one).
    *   It constructs an `A2ATask` object with necessary details:
        *   `taskId`: A unique ID for the task.
        *   `clientAgentId`: OpenAI agent's ID.
        *   `remoteAgentId`: Anthropic agent's ID.
        *   `capability_name`: The chosen capability of the Anthropic agent.
        *   `parameters`: Dummy parameters for the chosen capability.

4.  **Simulated A2A Call**:
    *   The script logs the intention to send the task.
    *   **Crucially, instead of making an actual HTTP network call (as the A2A server endpoints are not running within the agents themselves), the demo directly calls the `anthropic_agent_instance.handle_a2a_task_request(task_data)` method.** This simulates the Anthropic agent receiving and processing the task.
    *   The `handle_a2a_task_request` method in `AnthropicAgent` (and other agents) is a placeholder that currently logs the request and returns a mock JSON-RPC success or error response.

5.  **Response Processing**:
    *   The script logs the (simulated) response received from the `AnthropicAgent`.
    *   It checks if the response contains a "result" or an "error" and logs relevant details like task status and artifacts.

**Purpose of the Demo:**
*   To illustrate the flow of A2A communication: agent discovery (via cards), task creation, (simulated) task delegation, and response handling.
*   To test the `AgentCard` generation and the basic structure of `A2ATask` and the `handle_a2a_task_request` placeholders.
*   It provides a foundation for building a more complete A2A implementation with actual HTTP servers and clients.

**To Run the Demo:**
1.  Ensure Python is installed.
2.  Install dependencies: `pip install openai anthropic requests` (requests is included for future-proofing the A2AClient, though not actively used for calls in this demo). The `a2a-sdk` was added to `requirements.txt` earlier.
3.  Set the environment variables: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
4.  Run the script: `python a2a_collaboration_demo.py`.
```

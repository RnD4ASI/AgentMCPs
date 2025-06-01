import asyncio
import json
import logging
import os
import uuid

# Import agents and A2A protocol components
from openai_agent import OpenAIAgent # Assuming this is the correct import path
from anthropic_agent import AnthropicAgent # Assuming this is the correct import path
from a2a_protocol import A2ATask, AgentCard, A2AClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Dummy MCP configs for agent initialization (if needed by agents)
AGENT_CONFIG = {
    "streaming_mcp_config": {},
    "federated_mcp_config": {},
    "robust_mcp_config": {}
}

async def main():
    """
    Demonstrates A2A collaboration between OpenAIAgent and AnthropicAgent.
    """
    logging.info("--- A2A Collaboration Demo Script ---")

    if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
        logging.error("API keys for OpenAI and Anthropic must be set as environment variables.")
        logging.error("Skipping A2A collaboration demo.")
        return

    # 1. Initialize Agents
    try:
        openai_agent_instance = OpenAIAgent(
            api_key=OPENAI_API_KEY,
            agent_config=AGENT_CONFIG
        )
        anthropic_agent_instance = AnthropicAgent(
            api_key=ANTHROPIC_API_KEY,
            agent_config=AGENT_CONFIG
        )
        logging.info("OpenAI and Anthropic agents initialized.")
    except Exception as e:
        logging.error(f"Error initializing agents: {e}")
        return

    # 2. Get Agent Cards
    openai_agent_card = openai_agent_instance.get_agent_card()
    anthropic_agent_card = anthropic_agent_instance.get_agent_card()

    logging.info(f"OpenAI Agent Card: {openai_agent_card.to_json()}")
    logging.info(f"Anthropic Agent Card: {anthropic_agent_card.to_json()}")

    # 3. Define a task for Anthropic Agent to perform
    # Let's assume OpenAI agent wants Anthropic agent to use one of its capabilities.
    # We'll pick the first capability listed in Anthropic's agent card for this demo.
    if not anthropic_agent_card.capabilities:
        logging.warning("Anthropic agent has no capabilities listed in its card. Cannot proceed with demo task.")
        return

    # For this demo, let's assume Anthropic has a unique capability OpenAI wants to use.
    # We'll try to invoke a "RobustMCP_process_message" if available, or the first one.
    target_capability_name = "RobustMCP_process_message" # Example

    # Find the capability details from Anthropic's card
    chosen_capability = next((cap for cap in anthropic_agent_card.capabilities if cap['name'] == target_capability_name), None)
    if not chosen_capability:
        chosen_capability = anthropic_agent_card.capabilities[0] # Fallback to the first capability
        target_capability_name = chosen_capability['name']
        logging.info(f"Target capability '{target_capability_name}' not found, using first available: {target_capability_name}")


    # Construct parameters for the chosen capability
    # This is highly dependent on the capability's defined parameters.
    # For "RobustMCP_process_message", it might expect a 'message' parameter.
    # For a generic capability, parameters might be different.
    # We'll create some dummy parameters.
    task_parameters = {}
    if chosen_capability and chosen_capability.get('parameters'):
        for param_info in chosen_capability['parameters']:
            if param_info['type'] == 'string':
                task_parameters[param_info['name']] = f"Demo value for {param_info['name']}"
            elif param_info['type'] == 'object':
                task_parameters[param_info['name']] = {"sub_key": "sub_value_demo"}
            elif param_info['type'] == 'boolean':
                task_parameters[param_info['name']] = True
            else: # Default for other types
                task_parameters[param_info['name']] = "default_demo_param_value"
    else: # Fallback if no parameters are defined in the capability
        task_parameters = {"prompt": "Tell me a short story about AI collaboration."}


    a2a_task_to_anthropic = A2ATask(
        task_id="task-" + str(uuid.uuid4()),
        client_agent_id=openai_agent_card.agent_id,
        remote_agent_id=anthropic_agent_card.agent_id,
        capability_name=target_capability_name,
        parameters=task_parameters
    )

    logging.info(f"OpenAI Agent creating task for Anthropic Agent: {a2a_task_to_anthropic.to_json()}")

    # 4. Simulate OpenAI Agent sending the task to Anthropic Agent
    # In a real scenario, openai_agent_instance.a2a_client.send_task_request would make an HTTP call.
    # Here, we directly call the target agent's handler for demonstration purposes,
    # as the HTTP server part is not yet implemented for each agent.

    logging.info(f"Simulating A2A call from {openai_agent_card.name} to {anthropic_agent_card.name}...")

    # The A2AClient's send_task_request is currently mocked.
    # To make the demo more illustrative of the interaction, we will directly call
    # the handler on the anthropic_agent_instance.
    # This simulates the anthropic_agent receiving the request.

    # Convert task to dict as the handler expects dict
    task_data_for_anthropic = a2a_task_to_anthropic.to_dict()

    # Simulate the call to the remote agent's task handler
    # (This bypasses the network for the demo)
    if hasattr(anthropic_agent_instance, 'handle_a2a_task_request'):
        logging.info(f"Invoking {anthropic_agent_card.name}'s handle_a2a_task_request directly for demo.")
        response_from_anthropic = anthropic_agent_instance.handle_a2a_task_request(task_data_for_anthropic)
    else:
        logging.error(f"{anthropic_agent_card.name} does not have 'handle_a2a_task_request' method.")
        response_from_anthropic = {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found on agent instance for demo simulation"},
            "id": task_data_for_anthropic.get("id")
        }

    logging.info(f"Response received by OpenAI Agent from Anthropic Agent (simulated): {json.dumps(response_from_anthropic, indent=2)}")

    # 5. Process the response
    if "result" in response_from_anthropic:
        task_result = response_from_anthropic["result"]
        logging.info(f"Task '{task_result.get('taskId')}' status: {task_result.get('status')}")
        if task_result.get('artifacts'):
            logging.info("Artifacts received:")
            for artifact in task_result['artifacts']:
                logging.info(f"  - Type: {artifact.get('type')}, Content: {str(artifact.get('content'))[:200]}...")
    elif "error" in response_from_anthropic:
        task_error = response_from_anthropic["error"]
        logging.error(f"A2A Task failed: Code {task_error.get('code')}, Message: {task_error.get('message')}")

    logging.info("--- End of A2A Collaboration Demo ---")

if __name__ == "__main__":
    # Note: The A2AClient's send_task_request is async, but for this direct call demo,
    # we are calling the synchronous handle_a2a_task_request.
    # If we were using the actual a2a_client.send_task_request, we'd need asyncio.run(main())
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # This handles environments like Jupyter notebooks where an event loop is already running.
            # We create a task to run main() in the existing loop.
            logging.warning("Running in an existing event loop. Scheduling main() as a task.")
            loop = asyncio.get_event_loop()
            loop.create_task(main())
            # Note: In a script, if you reach here, you might need to explicitly run the loop
            # if it's not already running in a way that processes create_task.
            # However, for typical script execution, asyncio.run() is standard.
            # For Jupyter, this create_task approach is common.
        else:
            # Re-raise other RuntimeErrors
            raise

import os
import json
import importlib.util
import time
import logging
import uuid # Added for A2A agent_id
from openai import OpenAI # Standard OpenAI library

from a2a_protocol import AgentCard, A2ATask, A2AClient # A2A imports

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "openai_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "openai_agent_procedural_memory.json"
MCP_CONFIG_FILE = "mcp_configurations.json"

# Custom MCP module paths (assuming they are in the same directory)
CUSTOM_MCP_MODULES = {
    "RealTimeStreamingMCP": {
        "module_name": "custom_mcp_1",
        "class_name": "RealTimeStreamingMCP",
        "config_key": "streaming_mcp_config"
    },
    "FederatedLearningMCP": {
        "module_name": "custom_mcp_2",
        "class_name": "FederatedLearningMCP",
        "config_key": "federated_mcp_config"
    },
    "RobustMCP": {
        "module_name": "custom_mcp_3",
        "class_name": "RobustMCP",
        "config_key": "robust_mcp_config"
    }
}

class OpenAIAgent:
    """
    An AI agent that utilizes OpenAI services and custom MCP tools,
    with capabilities for managing factual and procedural memory.
    This agent is similar to AzureOpenAIAgent but uses the standard OpenAI API.
    """

    def __init__(self, api_key, model_name="gpt-3.5-turbo", agent_config=None):
        """
        Initializes the OpenAIAgent.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
            agent_config (dict, optional): Configuration for the agent, including MCP configs.
        """
        if not api_key:
            raise ValueError("OpenAI API key must be provided.")
        if not model_name:
            raise ValueError("OpenAI model name must be provided.")

        self.api_key = api_key
        self.model_name = model_name
        self.agent_config = agent_config if agent_config else {}

        try:
            self.openai_client = OpenAI(api_key=self.api_key)
            logging.info(f"OpenAI client initialized successfully for model {self.model_name}.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)
        
        self.mcp_configurations = self._load_mcp_configs(MCP_CONFIG_FILE)
        self.custom_mcps = self._load_custom_mcps()

        # A2A Initialization
        self.a2a_client = A2AClient()
        self.agent_id = f"openai-agent-{str(uuid.uuid4())}"
        self.a2a_endpoint_url = f"http://localhost:8001/a2a/{self.agent_id}" # Example endpoint
        logging.info(f"A2A Client initialized. Agent ID: {self.agent_id}, Endpoint: {self.a2a_endpoint_url}")

        logging.info("Agent initialized with memory, MCP tools, and A2A capabilities.")

    def _load_memory(self, filepath):
        """Loads memory from a JSON file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    memory = json.load(f)
                logging.info(f"Loaded memory from {filepath}.")
                return memory
            else:
                logging.info(f"Memory file {filepath} not found. Starting with empty memory.")
                return {}
        except Exception as e:
            logging.error(f"Error loading memory from {filepath}: {e}")
            return {}

    def _save_memory(self, filepath, data):
        """Saves memory to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved memory to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving memory to {filepath}: {e}")

    def add_factual_memory(self, fact_key, fact_value):
        """Adds or updates a fact in factual memory."""
        self.factual_memory[fact_key] = fact_value
        self._save_memory(FACTUAL_MEMORY_FILE, self.factual_memory)
        logging.info(f"Added/Updated fact: '{fact_key}' = '{fact_value}'")

    def get_factual_memory(self, fact_key, default=None):
        """Retrieves a fact from factual memory."""
        value = self.factual_memory.get(fact_key, default)
        logging.info(f"Retrieved fact: '{fact_key}' = '{value}'")
        return value

    def add_procedural_memory(self, task_name, steps):
        """Adds or updates a procedure (sequence of actions) in procedural memory."""
        self.procedural_memory[task_name] = {
            "steps": steps,
            "last_used_timestamp": time.time()
        }
        self._save_memory(PROCEDURAL_MEMORY_FILE, self.procedural_memory)
        logging.info(f"Added/Updated procedure: '{task_name}' with {len(steps)} steps.")

    def get_procedural_memory(self, task_name, default=None):
        """Retrieves a procedure from procedural memory."""
        procedure = self.procedural_memory.get(task_name, default)
        if procedure:
            logging.info(f"Retrieved procedure: '{task_name}'")
        else:
            logging.info(f"Procedure '{task_name}' not found.")
        return procedure

    def find_relevant_procedure(self, query_keywords):
        """
        Finds a relevant procedure based on keywords in the query.
        (Simple keyword matching for this version).
        """
        for task_name, procedure_data in self.procedural_memory.items():
            if any(keyword.lower() in task_name.lower() for keyword in query_keywords):
                logging.info(f"Found relevant procedure '{task_name}' for keywords: {query_keywords}")
                return procedure_data
        logging.info(f"No specific procedure found for keywords: {query_keywords}")
        return None

    def _load_mcp_configs(self, filepath):
        """Loads MCP configurations from the JSON file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    configs = json.load(f)
                logging.info(f"Loaded MCP configurations from {filepath}.")
                return configs
            else:
                logging.warning(f"MCP configuration file {filepath} not found.")
                return []
        except Exception as e:
            logging.error(f"Error loading MCP configurations from {filepath}: {e}")
            return []

    def _load_custom_mcps(self):
        """Dynamically loads and instantiates custom MCP classes."""
        mcps = {}
        for mcp_name, mcp_details in CUSTOM_MCP_MODULES.items():
            try:
                module_path = f"{mcp_details['module_name']}.py"
                spec = importlib.util.spec_from_file_location(mcp_details['module_name'], module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    mcp_class = getattr(module, mcp_details['class_name'])
                    
                    mcp_specific_config_key = mcp_details.get('config_key')
                    mcp_instance_config = self.agent_config.get(mcp_specific_config_key, {})

                    if mcp_details['class_name'] == "FederatedLearningMCP":
                        model_id = mcp_instance_config.get("global_model_id", "default_global_model")
                        mcps[mcp_name] = mcp_class(global_model_id=model_id)
                    else:
                        mcps[mcp_name] = mcp_class(config=mcp_instance_config)
                    logging.info(f"Successfully loaded and instantiated MCP: {mcp_name}")
                else:
                    logging.error(f"Could not create spec for module {mcp_details['module_name']} at {module_path}")
            except FileNotFoundError:
                logging.error(f"Custom MCP file {module_path} not found.")
            except AttributeError:
                logging.error(f"Class {mcp_details['class_name']} not found in module {mcp_details['module_name']}.")
            except Exception as e:
                logging.error(f"Failed to load custom MCP {mcp_name}: {e}")
        return mcps

    def _get_openai_chat_completion(self, messages, max_tokens=150, temperature=0.7):
        """
        Gets a chat completion from OpenAI.

        Args:
            messages (list): A list of message objects (e.g., [{"role": "user", "content": "Hello"}]).
            max_tokens (int): Maximum number of tokens for the completion.
            temperature (float): Sampling temperature.

        Returns:
            str: The completion text, or an error message.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logging.info(f"OpenAI Chat API call successful. Model: {self.model_name}, Messages: {messages}")
            return response.choices[0].message.content.strip()
        except Exception as e: # Catching openai.APIError or other specific errors can be more precise
            logging.error(f"OpenAI Chat API call failed: {e}")
            return f"Error communicating with OpenAI: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """
        Uses LLM to determine user intent, required MCP, and parameters.
        (Identical to AzureOpenAIAgent's method, but calls self._get_openai_chat_completion)
        """
        mcp_tool_descriptions = []
        for name, mcp_instance in self.custom_mcps.items():
            mcp_tool_descriptions.append(f"- {name}: {mcp_instance.__doc__.splitlines()[0].strip() if mcp_instance.__doc__ else 'No description'}")
        
        mcp_config_descriptions = []
        for config in self.mcp_configurations: # Show a few examples
            mcp_config_descriptions.append(f"- Name: {config.get('name')}, Description: {config.get('description')}, Protocol: {config.get('protocol_details', {}).get('type')}")

        system_prompt = f"""
You are an AI assistant that helps users by leveraging a set of available tools (MCPs - Model Context Protocols)
and pre-defined MCP configurations. Your goal is to understand the user's query, identify if a specific
MCP tool or a general MCP configuration is needed, and extract necessary parameters.

Available Custom MCP Tools:
{json.dumps(CUSTOM_MCP_MODULES, indent=2)}

Available Pre-defined MCP Configurations (from mcp_configurations.json, showing first 2 for brevity):
{json.dumps(self.mcp_configurations[:2], indent=2)} 

Respond in JSON format with the following fields:
- "intent": A brief description of what the user wants to do.
- "mcp_tool_name": The name of the custom MCP tool to use (e.g., "RealTimeStreamingMCP", "FederatedLearningMCP", "RobustMCP"), or "None" if no specific custom tool is directly applicable or if a generic configuration is more suitable.
- "mcp_config_name": The name of a pre-defined MCP configuration to use from mcp_configurations.json (e.g., "Standard HTTPS/JSON MCP"), or "None".
- "parameters": A dictionary of parameters needed for the MCP or to fulfill the query. For custom MCPs, these are parameters for their methods.
- "requires_further_clarification": boolean, true if you cannot determine the above with confidence.
- "clarification_question": string, a question to ask the user if requires_further_clarification is true.
- "response_to_user": A direct, friendly response to the user if no tool is needed or if clarification is needed.

Example Query: "I need to process a stream of sensor data for anomalies."
Example JSON Response:
{{
  "intent": "Process real-time sensor data for anomaly detection",
  "mcp_tool_name": "RealTimeStreamingMCP",
  "mcp_config_name": null,
  "parameters": {{ "raw_data_json": "User needs to provide this or specify source" }},
  "requires_further_clarification": true,
  "clarification_question": "Could you please provide the sensor data stream or tell me where to get it?",
  "response_to_user": "I can help with that. I'll use the RealTimeStreamingMCP. Could you please provide the sensor data stream or tell me where to get it?"
}}

If the query is about past actions or stored information, try to use that from memory.
Factual Memory available (keys only): {list(self.factual_memory.keys())}
Procedural Memory available (task names only): {list(self.procedural_memory.keys())}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        llm_response_str = self._get_openai_chat_completion(messages=messages, max_tokens=500, temperature=0.2)
        
        try:
            parsed_response = json.loads(llm_response_str)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response as JSON: {llm_response_str}")
            return {
                "intent": "Error in understanding query", 
                "mcp_tool_name": None, 
                "mcp_config_name": None,
                "parameters": {}, 
                "requires_further_clarification": True,
                "clarification_question": "I had trouble understanding your request. Could you please rephrase?",
                "response_to_user": "I'm having a little trouble understanding. Could you try asking in a different way?"
            }

    def use_custom_mcp(self, mcp_name, method_name, params):
        """
        Invokes a method on a specified custom MCP instance.
        (Identical to AzureOpenAIAgent's method)
        """
        if mcp_name not in self.custom_mcps:
            logging.error(f"Custom MCP '{mcp_name}' not found.")
            return f"Error: Custom MCP tool '{mcp_name}' is not available."

        mcp_instance = self.custom_mcps[mcp_name]
        if not hasattr(mcp_instance, method_name):
            logging.error(f"Method '{method_name}' not found in MCP '{mcp_name}'.")
            return f"Error: Method '{method_name}' does not exist for MCP '{mcp_name}'."

        resolved_params = {}
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("GET_FROM_FACTUAL_MEMORY:"):
                    mem_key = value.split(":", 1)[1]
                    resolved_params[key] = self.get_factual_memory(mem_key)
                    if resolved_params[key] is None:
                        logging.warning(f"Memory key {mem_key} not found for MCP parameter {key}")
                        return f"Error: Required information '{mem_key}' not found in memory for MCP {mcp_name}."
                else:
                    resolved_params[key] = value
        elif params is not None:
             resolved_params = params
        
        try:
            method_to_call = getattr(mcp_instance, method_name)
            logging.info(f"Calling MCP '{mcp_name}', method '{method_name}' with params: {resolved_params}")
            
            if isinstance(resolved_params, dict):
                result = method_to_call(**resolved_params)
            elif resolved_params is not None:
                result = method_to_call(resolved_params)
            else:
                result = method_to_call()

            logging.info(f"MCP '{mcp_name}' method '{method_name}' executed. Result: {str(result)[:100]}...")
            self.add_procedural_memory(
                task_name=f"Executed {mcp_name}.{method_name}",
                steps=[f"Called {method_name} on {mcp_name} with {params}", f"Result: {str(result)[:100]}..."]
            )
            return result if result is not None else "MCP method executed, no specific result returned."
        except Exception as e:
            logging.error(f"Error executing MCP '{mcp_name}' method '{method_name}': {e}")
            return f"Error during MCP execution: {str(e)}"

    def process_query(self, query):
        """
        Processes a user query, determines intent, uses tools, and manages memory.
        (Identical logic to AzureOpenAIAgent's method, but calls the OpenAI LLM)
        """
        logging.info(f"\n--- Processing Query: '{query}' ---")

        intent_data = self._determine_intent_and_params_with_llm(query)
        response_to_user = intent_data.get("response_to_user", "I'm processing your request.")
        
        if intent_data.get("requires_further_clarification"):
            self.add_factual_memory("last_clarification_for_query", query)
            self.add_factual_memory("last_clarification_question", intent_data.get("clarification_question"))
            return intent_data.get("clarification_question", response_to_user)

        mcp_tool_name = intent_data.get("mcp_tool_name")
        mcp_config_name = intent_data.get("mcp_config_name")
        params = intent_data.get("parameters", {})

        if mcp_tool_name and mcp_tool_name != "None":
            default_methods = {
                "RealTimeStreamingMCP": "process_stream_data",
                "FederatedLearningMCP": "get_global_model",
                "RobustMCP": "process_message"
            }
            method_to_call = params.pop("method_name", default_methods.get(mcp_tool_name))

            if not method_to_call:
                response_to_user = f"I identified {mcp_tool_name} but I'm unsure which action to perform. Can you be more specific?"
                self.add_procedural_memory(f"Query for {mcp_tool_name}", [query, f"Uncertain on method for {mcp_tool_name}"])
            else:
                mcp_result = self.use_custom_mcp(mcp_name=mcp_tool_name, method_name=method_to_call, params=params.get("mcp_params", params))
                response_to_user = f"Used {mcp_tool_name}: {str(mcp_result)[:200]}..."
                self.add_factual_memory(f"last_mcp_result_{mcp_tool_name}", str(mcp_result)[:500])
        
        elif mcp_config_name and mcp_config_name != "None":
            selected_config = next((c for c in self.mcp_configurations if c.get("name") == mcp_config_name), None)
            if selected_config:
                response_to_user = f"Found MCP configuration: '{mcp_config_name}'. Details: {selected_config['protocol_details']}. How would you like to use this?"
                self.add_factual_memory("last_selected_mcp_config", selected_config)
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Identified config: {mcp_config_name}"])
            else:
                response_to_user = f"Could not find details for MCP configuration: '{mcp_config_name}'."
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Failed to find config: {mcp_config_name}"])
        
        elif not (mcp_tool_name and mcp_tool_name != "None") and not (mcp_config_name and mcp_config_name != "None") and intent_data.get("response_to_user"):
             response_to_user = intent_data.get("response_to_user")
             self.add_procedural_memory("General query", [query, f"LLM response: {response_to_user}"])

        if not mcp_tool_name and not mcp_config_name and not intent_data.get("requires_further_clarification") and response_to_user == "I'm processing your request.":
            logging.info("No specific tool or config identified by LLM, and no direct response. Using general LLM completion.")
            response_to_user = self._get_openai_chat_completion(
                 messages=[{"role": "user", "content": query}],
                 max_tokens=100
            )
            self.add_procedural_memory("General query fallback", [query, f"LLM fallback response: {response_to_user}"])

        logging.info(f"Final Response: {response_to_user}")
        return response_to_user

    def get_agent_card(self) -> AgentCard:
        """Constructs and returns the AgentCard for this agent."""
        capabilities = []
        for mcp_name, mcp_instance in self.custom_mcps.items():
            docstring = mcp_instance.__doc__.splitlines()[0].strip() if mcp_instance.__doc__ else f"Default capability for {mcp_name}"

            # Simplified parameter and return definition
            params_schema = []
            default_methods = {
                "RealTimeStreamingMCP": "process_stream_data",
                "FederatedLearningMCP": "get_global_model", # Assuming this is a primary action
                "RobustMCP": "process_message" # Assuming this is a primary action
            }
            method_name = default_methods.get(mcp_name, "execute") # Default to 'execute' if not mapped

            if mcp_name == "RealTimeStreamingMCP" and hasattr(mcp_instance, "process_stream_data"):
                # Specific for RealTimeStreamingMCP based on its typical usage
                params_schema.append(
                    {"name": "raw_data_json", "type": "string", "required": True, "description": "JSON string of the raw data stream."}
                )
                capability_name = f"{mcp_name}_process_stream_data"
                description = f"Processes real-time streaming data using {mcp_name}."
            elif mcp_name == "FederatedLearningMCP" and hasattr(mcp_instance, "get_global_model"):
                 capability_name = f"{mcp_name}_get_global_model"
                 description = f"Retrieves the global model using {mcp_name}."
                 # No specific parameters needed for get_global_model based on its current implementation
            elif mcp_name == "RobustMCP" and hasattr(mcp_instance, "process_message"):
                params_schema.append(
                    {"name": "message", "type": "object", "required": True, "description": "The message object to process."}
                )
                capability_name = f"{mcp_name}_process_message"
                description = f"Processes a message robustly using {mcp_name}."
            else:
                # Generic capability if specific method not detailed above
                capability_name = f"{mcp_name}_{method_name}"
                description = docstring
                params_schema.append(
                    {"name": "params", "type": "object", "required": False, "description": f"Parameters for {method_name} of {mcp_name}."}
                )

            capabilities.append({
                "name": capability_name,
                "description": description,
                "parameters": params_schema,
                "returns": {"type": "object", "description": f"Result of {capability_name} execution."}
            })

        return AgentCard(
            agent_id=self.agent_id,
            name="OpenAI Assistant Agent",
            description="An agent powered by OpenAI, capable of general queries and using specialized MCP tools.",
            version="1.0.0",
            capabilities=capabilities,
            endpoint_url=self.a2a_endpoint_url,
            documentation_url=None # Optional: Add a link to documentation if available
        )

    def handle_a2a_task_request(self, task_data: dict) -> dict:
        """
        Handles an incoming A2A task request.
        Simulates task execution for now.
        """
        logging.info(f"Received A2A task request: {json.dumps(task_data, indent=2)}")
        try:
            task = A2ATask.from_dict(task_data)
            logging.info(f"Parsed A2A Task: {task.to_json()}")

            agent_capabilities = self.get_agent_card().capabilities
            is_capable = any(cap['name'] == task.capability_name for cap in agent_capabilities)

            if is_capable:
                logging.info(f"Simulating execution of A2A capability: {task.capability_name} with params: {task.parameters}")
                # TODO: Implement actual task execution logic.
                # This might involve calling self.use_custom_mcp or other internal methods.
                # Example:
                # mcp_name, method_name = task.capability_name.split('_', 1)
                # if mcp_name in self.custom_mcps and hasattr(self.custom_mcps[mcp_name], method_name):
                #     mcp_result = self.use_custom_mcp(mcp_name, method_name, task.parameters)
                #     return {
                #         "jsonrpc": "2.0",
                #         "result": {
                #             "taskId": task.task_id,
                #             "status": "completed",
                #             "artifacts": [{"type": "json", "content": mcp_result}]
                #         },
                #         "id": task_data.get("id")
                #     }

                # Mock success response
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "taskId": task.task_id,
                        "status": "completed",
                        "artifacts": [{"type": "text", "content": f"Successfully executed A2A capability: {task.capability_name}"}]
                    },
                    "id": task_data.get("id")
                }
            else:
                logging.warning(f"Capability '{task.capability_name}' not found for A2A task {task.task_id}.")
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found", "data": f"Capability '{task.capability_name}' not implemented by agent {self.agent_id}."},
                    "id": task_data.get("id")
                }
        except Exception as e:
            logging.error(f"Error handling A2A task: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": task_data.get("id") # Ensure ID is returned even in case of parsing error if possible
            }

# Example Usage
if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable must be set.")
        print("Skipping OpenAIAgent demo.")
    else:
        print("--- OpenAIAgent Demo ---")

        # Agent-specific configurations for its MCPs
        agent_specific_mcp_configs = {
            "streaming_mcp_config": {
                'validation_rules': {
                    'sensor_id': {'type': str},
                    'value': {'type': float, 'min': 0.0, 'max': 100.0}
                },
                'transformation_map': {} 
            },
            "federated_mcp_config": {
                "global_model_id": "openai_agent_demo_model_v1"
            },
            "robust_mcp_config": {
                'max_retries': 1,
                'base_retry_delay_seconds': 0.1,
            }
        }

        try:
            agent = OpenAIAgent(
                api_key=OPENAI_API_KEY,
                model_name=OPENAI_MODEL_NAME,
                agent_config=agent_specific_mcp_configs
            )

            agent.add_factual_memory("user_language_preference", "English")
            print(f"User's language preference: {agent.get_factual_memory('user_language_preference')}")

            agent.add_procedural_memory("agent_boot_sequence", ["Initialized client", "Loaded memory", "Loaded MCPs"])
            print(f"Agent boot sequence: {agent.get_procedural_memory('agent_boot_sequence')}")

            queries = [
                "Hi there, what are your capabilities?",
                "I need to process some real-time sensor data. The data is: '[{\"sensor_id\": \"s001\", \"value\": 75.2}]'",
                "What was the last result from the streaming MCP?",
                "How can I robustly send a message like '{\"id\": \"msg789\", \"payload\": {\"data\": \"test_data\", \"force_failure_type\": \"permanent\"}}'?",
                "Find information about the 'High-Performance gRPC MCP' configuration.",
                "My favorite color is blue." # LLM should store this if prompted correctly or handled by intent parser
            ]

            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                time.sleep(2) # Adhere to rate limits and for readability

            print("\n--- Final Memory States ---")
            print(f"Factual Memory: {json.dumps(agent.factual_memory, indent=2)}")
            print(f"Procedural Memory: {json.dumps(agent.procedural_memory, indent=2)}")

            print("\n--- Agent Card ---")
            agent_card = agent.get_agent_card()
            print(agent_card.to_json())

            print("\n--- A2A Task Handling Simulation ---")
            # Simulate an incoming A2A task request
            sample_a2a_task_data_valid = {
                "jsonrpc": "2.0",
                "method": "executeTask", # This is illustrative; actual method might vary
                "params": {
                    "taskId": "a2a-task-example-001",
                    "clientAgentId": "test-client-agent",
                    "remoteAgentId": agent.agent_id, # Targeting this agent
                    "capabilityName": "RealTimeStreamingMCP_process_stream_data",
                    "parameters": {"raw_data_json": "[{\"sensor_id\":\"s789\", \"value\":99.0}]"}
                },
                "id": "rpc-call-123"
            }
            a2a_response = agent.handle_a2a_task_request(sample_a2a_task_data_valid["params"]) # Pass the task part
            print(f"A2A Task Response (Valid Task): {json.dumps(a2a_response, indent=2)}")

            sample_a2a_task_data_invalid_capability = {
                 "jsonrpc": "2.0",
                 "method": "executeTask",
                 "params": {
                    "taskId": "a2a-task-example-002",
                    "clientAgentId": "test-client-agent",
                    "remoteAgentId": agent.agent_id,
                    "capabilityName": "NonExistentCapability_do_something",
                    "parameters": {}
                 },
                 "id": "rpc-call-456"
            }
            a2a_response_invalid = agent.handle_a2a_task_request(sample_a2a_task_data_invalid_capability["params"])
            print(f"A2A Task Response (Invalid Capability): {json.dumps(a2a_response_invalid, indent=2)}")


        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during the demo: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- End of OpenAIAgent Demo ---")

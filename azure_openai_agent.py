import os
import json
import importlib.util
import time
import logging
from openai import AzureOpenAI

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "azure_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "azure_agent_procedural_memory.json"
MCP_CONFIG_FILE = "mcp_configurations.json"

# Custom MCP module paths (assuming they are in the same directory)
CUSTOM_MCP_MODULES = {
    "RealTimeStreamingMCP": {
        "module_name": "custom_mcp_1",
        "class_name": "RealTimeStreamingMCP",
        "config_key": "streaming_mcp_config" # Key for specific config in agent's own config
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

class AzureOpenAIAgent:
    """
    An AI agent that utilizes Azure OpenAI services and custom MCP tools,
    with capabilities for managing factual and procedural memory.
    """

    def __init__(self, azure_api_key, azure_endpoint, azure_api_version="2023-12-01-preview", llm_deployment_name="gpt-35-turbo", agent_config=None):
        """
        Initializes the AzureOpenAIAgent.

        Args:
            azure_api_key (str): Azure OpenAI API key.
            azure_endpoint (str): Azure OpenAI endpoint URL.
            azure_api_version (str): Azure OpenAI API version.
            llm_deployment_name (str): The name of the LLM deployment on Azure.
            agent_config (dict, optional): Configuration for the agent, including MCP configs.
        """
        if not all([azure_api_key, azure_endpoint, llm_deployment_name]):
            raise ValueError("Azure API key, endpoint, and LLM deployment name must be provided.")

        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.llm_deployment_name = llm_deployment_name
        self.agent_config = agent_config if agent_config else {}

        try:
            self.openai_client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint
            )
            logging.info("Azure OpenAI client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)
        
        self.mcp_configurations = self._load_mcp_configs(MCP_CONFIG_FILE)
        self.custom_mcps = self._load_custom_mcps()
        logging.info("Agent initialized with memory and MCP tools.")

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
                    
                    # Get specific config for this MCP from agent_config if available
                    mcp_specific_config_key = mcp_details.get('config_key')
                    mcp_instance_config = self.agent_config.get(mcp_specific_config_key, {})

                    if mcp_details['class_name'] == "FederatedLearningMCP": # Requires specific args
                        # Example: provide a default model ID or get from agent_config
                        model_id = mcp_instance_config.get("global_model_id", "default_global_model")
                        mcps[mcp_name] = mcp_class(global_model_id=model_id)
                    else: # Assumes other MCPs take a single 'config' dict
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

    def _get_openai_completion(self, prompt, max_tokens=150, temperature=0.7):
        """
        Gets a completion from Azure OpenAI.

        Args:
            prompt (str): The prompt to send to the LLM.
            max_tokens (int): Maximum number of tokens for the completion.
            temperature (float): Sampling temperature.

        Returns:
            str: The completion text, or an error message.
        """
        try:
            response = self.openai_client.completions.create(
                model=self.llm_deployment_name, # This should be your deployment name for a completion model
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logging.info(f"Azure OpenAI API call successful. Prompt: '{prompt[:50]}...'")
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error(f"Azure OpenAI API call failed: {e}")
            if "DeploymentNotFound" in str(e) or "ResourceNotFound" in str(e):
                 return f"Error: LLM Deployment '{self.llm_deployment_name}' not found. Please check the deployment name and API configuration."
            return f"Error communicating with Azure OpenAI: {str(e)}"

    def _get_openai_chat_completion(self, messages, max_tokens=150, temperature=0.7):
        """
        Gets a chat completion from Azure OpenAI (for models like GPT-3.5-Turbo and GPT-4).

        Args:
            messages (list): A list of message objects (e.g., [{"role": "user", "content": "Hello"}]).
            max_tokens (int): Maximum number of tokens for the completion.
            temperature (float): Sampling temperature.

        Returns:
            str: The completion text, or an error message.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_deployment_name, # This should be your deployment name for a chat model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logging.info(f"Azure OpenAI Chat API call successful. Messages: {messages}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Azure OpenAI Chat API call failed: {e}")
            if "DeploymentNotFound" in str(e) or "ResourceNotFound" in str(e):
                 return f"Error: LLM Deployment '{self.llm_deployment_name}' not found. Please check the deployment name and API configuration."
            return f"Error communicating with Azure OpenAI (Chat): {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """
        Uses LLM to determine user intent, required MCP, and parameters.
        This is a more advanced approach than simple keyword matching.
        """
        mcp_tool_descriptions = []
        for name, mcp_instance in self.custom_mcps.items():
            mcp_tool_descriptions.append(f"- {name}: {mcp_instance.__doc__.splitlines()[0].strip() if mcp_instance.__doc__ else 'No description'}")
        
        mcp_config_descriptions = []
        for config in self.mcp_configurations:
            mcp_config_descriptions.append(f"- Name: {config.get('name')}, Description: {config.get('description')}, Protocol: {config.get('protocol_details', {}).get('type')}")

        system_prompt = f"""
You are an AI assistant that helps users by leveraging a set of available tools (MCPs - Model Context Protocols)
and pre-defined MCP configurations. Your goal is to understand the user's query, identify if a specific
MCP tool or a general MCP configuration is needed, and extract necessary parameters.

Available Custom MCP Tools:
{json.dumps(CUSTOM_MCP_MODULES, indent=2)}

Available Pre-defined MCP Configurations (from mcp_configurations.json):
{json.dumps(self.mcp_configurations[:2], indent=2)} (Showing first 2 for brevity)

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
            # Fallback or error handling
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

        Args:
            mcp_name (str): The name of the custom MCP tool (e.g., "RealTimeStreamingMCP").
            method_name (str): The method to call on the MCP instance.
            params (dict): A dictionary of parameters to pass to the MCP method. 
                           If a parameter is a string "GET_FROM_FACTUAL_MEMORY:<key>",
                           it will be fetched from factual memory.

        Returns:
            The result from the MCP method, or an error message.
        """
        if mcp_name not in self.custom_mcps:
            logging.error(f"Custom MCP '{mcp_name}' not found.")
            return f"Error: Custom MCP tool '{mcp_name}' is not available."

        mcp_instance = self.custom_mcps[mcp_name]
        if not hasattr(mcp_instance, method_name):
            logging.error(f"Method '{method_name}' not found in MCP '{mcp_name}'.")
            return f"Error: Method '{method_name}' does not exist for MCP '{mcp_name}'."

        # Resolve parameters from factual memory if specified
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
        elif params is not None: # if params is not a dict but also not None (e.g. a single string for some methods)
             resolved_params = params


        try:
            method_to_call = getattr(mcp_instance, method_name)
            logging.info(f"Calling MCP '{mcp_name}', method '{method_name}' with params: {resolved_params}")
            
            # Note: The custom MCPs expect specific argument structures.
            # This is a simplified invocation. Real implementation might need more sophisticated arg mapping.
            if isinstance(resolved_params, dict):
                result = method_to_call(**resolved_params)
            elif resolved_params is not None: # for methods that take a single non-dict argument
                result = method_to_call(resolved_params)
            else: # for methods that take no arguments
                result = method_to_call()

            logging.info(f"MCP '{mcp_name}' method '{method_name}' executed. Result: {str(result)[:100]}...") # Log snippet of result
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

        Args:
            query (str): The user's query.

        Returns:
            str: A response to the user.
        """
        logging.info(f"\n--- Processing Query: '{query}' ---")

        # 1. Understand Intent and Parameters using LLM
        intent_data = self._determine_intent_and_params_with_llm(query)
        
        response_to_user = intent_data.get("response_to_user", "I'm processing your request.")
        
        if intent_data.get("requires_further_clarification"):
            self.add_factual_memory("last_clarification_for_query", query)
            self.add_factual_memory("last_clarification_question", intent_data.get("clarification_question"))
            return intent_data.get("clarification_question", response_to_user)

        mcp_tool_name = intent_data.get("mcp_tool_name")
        mcp_config_name = intent_data.get("mcp_config_name")
        params = intent_data.get("parameters", {})

        # 2. Execute MCP tool if identified
        if mcp_tool_name and mcp_tool_name != "None":
            # This is a simplification. The LLM needs to also suggest the method.
            # For now, let's assume a default method or try to infer.
            # Example: RealTimeStreamingMCP might have 'process_stream_data'
            # This part needs refinement for robust method dispatch.
            default_methods = {
                "RealTimeStreamingMCP": "process_stream_data",
                "FederatedLearningMCP": "get_global_model", # Or based on params, e.g. "submit_model_update"
                "RobustMCP": "process_message"
            }
            method_to_call = params.pop("method_name", default_methods.get(mcp_tool_name))

            if not method_to_call:
                response_to_user = f"I identified {mcp_tool_name} but I'm unsure which action to perform with it. Can you be more specific?"
                self.add_procedural_memory(f"Query for {mcp_tool_name}", [query, f"Uncertain on method for {mcp_tool_name}"])
            else:
                # The parameters from LLM might need to be structured correctly for the method
                # e.g. process_stream_data expects 'raw_data_json'
                # e.g. submit_model_update expects 'participant_id', 'round_id', 'model_update_json'
                # This mapping is crucial and complex.
                mcp_result = self.use_custom_mcp(mcp_name=mcp_tool_name, method_name=method_to_call, params=params.get("mcp_params", params)) # pass params directly or nested under mcp_params
                response_to_user = f"Used {mcp_tool_name}: {str(mcp_result)[:200]}..."
                self.add_factual_memory(f"last_mcp_result_{mcp_tool_name}", str(mcp_result)[:500]) # Store snippet of result
        
        elif mcp_config_name and mcp_config_name != "None":
            selected_config = next((c for c in self.mcp_configurations if c.get("name") == mcp_config_name), None)
            if selected_config:
                response_to_user = f"Found MCP configuration: '{mcp_config_name}'. Details: {selected_config['protocol_details']}. How would you like to use this?"
                self.add_factual_memory("last_selected_mcp_config", selected_config)
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Identified config: {mcp_config_name}"])
            else:
                response_to_user = f"Could not find details for MCP configuration: '{mcp_config_name}'."
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Failed to find config: {mcp_config_name}"])
        
        # If no specific tool/config was used, but LLM provided a direct response
        elif not (mcp_tool_name and mcp_tool_name != "None") and not (mcp_config_name and mcp_config_name != "None") and intent_data.get("response_to_user"):
             response_to_user = intent_data.get("response_to_user")
             self.add_procedural_memory("General query", [query, f"LLM response: {response_to_user}"])


        # 3. Fallback or General LLM interaction if no tool was used and no direct response from intent parsing
        if not mcp_tool_name and not mcp_config_name and not intent_data.get("requires_further_clarification") and not intent_data.get("response_to_user"):
            logging.info("No specific tool or config identified. Using general LLM completion for response.")
            # This could be a simple echo, or a more sophisticated chat interaction
            # For now, we use the initial response_to_user which might have been set by the LLM.
            # If not, we can make another call.
            if response_to_user == "I'm processing your request.": # Default value if LLM didn't give one
                 response_to_user = self._get_openai_chat_completion(
                     messages=[{"role": "user", "content": query}],
                     max_tokens=100
                 )
                 self.add_procedural_memory("General query", [query, f"LLM fallback response: {response_to_user}"])


        logging.info(f"Final Response: {response_to_user}")
        return response_to_user

# Example Usage (Illustrative - requires Azure credentials and custom MCP files)
if __name__ == '__main__':
    AZURE_OAI_KEY = os.environ.get("AZURE_OAI_KEY")
    AZURE_OAI_ENDPOINT = os.environ.get("AZURE_OAI_ENDPOINT")
    AZURE_OAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OAI_LLM_DEPLOYMENT", "gpt-35-turbo") # Or your specific chat model deployment

    if not AZURE_OAI_KEY or not AZURE_OAI_ENDPOINT:
        print("Error: AZURE_OAI_KEY and AZURE_OAI_ENDPOINT environment variables must be set.")
        print("Skipping AzureOpenAIAgent demo.")
    else:
        print("--- AzureOpenAIAgent Demo ---")
        
        # Dummy config for custom MCPs (as an example)
        agent_specific_mcp_configs = {
            "streaming_mcp_config": {
                'validation_rules': {
                    'sensor_id': {'type': str},
                    'value': {'type': float, 'min': 0.0, 'max': 100.0}
                },
                'transformation_map': {} 
            },
            "federated_mcp_config": {
                "global_model_id": "agent_demo_model_v1"
            },
            "robust_mcp_config": {
                'max_retries': 1, # For faster demo
                'base_retry_delay_seconds': 0.1,
            }
        }

        try:
            agent = AzureOpenAIAgent(
                azure_api_key=AZURE_OAI_KEY,
                azure_endpoint=AZURE_OAI_ENDPOINT,
                llm_deployment_name=AZURE_OAI_LLM_DEPLOYMENT,
                agent_config=agent_specific_mcp_configs
            )

            # Test Factual Memory
            agent.add_factual_memory("user_preference_data_format", "JSON")
            print(f"User's preferred format: {agent.get_factual_memory('user_preference_data_format')}")

            # Test Procedural Memory
            agent.add_procedural_memory("initial_setup_task", ["Checked credentials", "Loaded MCPs", "Initialized memory"])
            print(f"Setup procedure: {agent.get_procedural_memory('initial_setup_task')}")

            # Example Queries
            queries = [
                "Hello, what can you do?",
                "I have some real-time sensor data that needs processing. It's in JSON format: '[{\"sensor_id\": \"temp01\", \"value\": 25.5}]'",
                "What was the last thing I asked you to do with sensor data?", # Tests memory recall via LLM
                "Can you show me a robust way to send a message? The message is '{\"id\": \"test001\", \"payload\": {\"data\": \"important_payload\", \"force_failure_type\": \"transient\"}}'",
                "Tell me about the standard HTTPS JSON MCP configuration.",
                "What are the steps for federated learning model aggregation?", # Tries to get procedural info via LLM
                "My name is Bob and I like XML." # Simple fact storing via LLM
            ]

            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                time.sleep(2) # To avoid hitting rate limits if any and for readability

            print("\n--- Final Memory States ---")
            print(f"Factual Memory: {json.dumps(agent.factual_memory, indent=2)}")
            print(f"Procedural Memory: {json.dumps(agent.procedural_memory, indent=2)}")

        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during the demo: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- End of AzureOpenAIAgent Demo ---")

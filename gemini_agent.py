import os
import json
import importlib.util
import time
import logging
import google.generativeai as genai # Google Gemini library

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "gemini_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "gemini_agent_procedural_memory.json"
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

class GeminiAgent:
    """
    An AI agent that utilizes Google Gemini services and custom MCP tools,
    with capabilities for managing factual and procedural memory.
    This agent is similar to previous agents but uses the Google Gemini API.
    """

    def __init__(self, api_key, model_name="gemini-pro", agent_config=None): # gemini-pro is a common model
        """
        Initializes the GeminiAgent.

        Args:
            api_key (str): Google Gemini API key.
            model_name (str): The name of the Gemini model to use (e.g., "gemini-pro", "gemini-1.5-flash").
            agent_config (dict, optional): Configuration for the agent, including MCP configs.
        """
        if not api_key:
            raise ValueError("Google Gemini API key must be provided.")
        if not model_name:
            raise ValueError("Google Gemini model name must be provided.")

        self.api_key = api_key
        self.model_name = model_name
        self.agent_config = agent_config if agent_config else {}

        try:
            genai.configure(api_key=self.api_key)
            # For safety settings, can be more granular:
            # self.safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # ]
            # self.gemini_model = genai.GenerativeModel(model_name=self.model_name, safety_settings=self.safety_settings)
            self.gemini_model = genai.GenerativeModel(model_name=self.model_name)
            logging.info(f"Google Gemini client initialized successfully for model {self.model_name}.")
        except Exception as e:
            logging.error(f"Failed to initialize Google Gemini client: {e}")
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
        """Finds a relevant procedure based on keywords in the query."""
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

    def _get_gemini_completion(self, prompt_text, max_output_tokens=1024, temperature=0.1):
        """
        Gets a completion from Google Gemini.

        Args:
            prompt_text (str): The prompt text to send to the Gemini model.
            max_output_tokens (int): Maximum number of tokens for the completion.
            temperature (float): Sampling temperature.

        Returns:
            str: The completion text, or an error message.
        """
        try:
            # Configuration for generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
            # Gemini API's generate_content can take a string or list of parts
            response = self.gemini_model.generate_content(
                prompt_text,
                generation_config=generation_config
                # stream=False # For non-streaming response
            )
            logging.info(f"Google Gemini API call successful. Model: {self.model_name}")
            
            # Check for blocked prompt or finish reason
            if not response.candidates or response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                logging.error(f"Gemini API call blocked. Reason: {block_reason}")
                safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.prompt_feedback.safety_ratings]) if response.prompt_feedback.safety_ratings else "N/A"
                return f"Error: Gemini API call blocked due to '{block_reason}'. Safety ratings: [{safety_ratings_str}]"

            # Accessing the text content:
            # response.text is a helper, but actual content is in response.parts
            if response.text:
                 return response.text
            elif response.parts:
                return "".join(part.text for part in response.parts if hasattr(part, "text"))
            else:
                logging.error("Gemini API response format not as expected or empty content.")
                return "Error: Received unexpected response format or empty content from Gemini."

        except Exception as e:
            logging.error(f"Google Gemini API call failed: {e}")
            return f"Error communicating with Google Gemini: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """
        Uses LLM to determine user intent, required MCP, and parameters.
        Adapted for Google Gemini.
        """
        # Gemini models can be instructed with a detailed prompt.
        # System-level instructions can be part of the main prompt.
        prompt = f"""
You are an AI assistant. Your task is to understand a user's query and determine if it can be addressed using one of your available tools (MCPs - Model Context Protocols) or pre-defined MCP configurations.
You must respond ONLY in a structured JSON format. Do not include any markdown specifiers like ```json ... ```.

Here are the available custom MCP tools:
<tools_description>
{json.dumps(CUSTOM_MCP_MODULES, indent=2)}
</tools_description>

Here are some available pre-defined MCP configurations (from mcp_configurations.json, showing first 2 for brevity):
<configurations_description>
{json.dumps(self.mcp_configurations[:2], indent=2)} 
</configurations_description>

Here is the factual memory you have access to (keys only):
<factual_memory_keys>
{list(self.factual_memory.keys())}
</factual_memory_keys>

Here is the procedural memory you have access to (task names only):
<procedural_memory_keys>
{list(self.procedural_memory.keys())}
</procedural_memory_keys>

Based on the user's query, you need to output a JSON object with the following fields:
- "intent": A brief description of what the user wants to do.
- "mcp_tool_name": The name of the custom MCP tool to use (e.g., "RealTimeStreamingMCP", "FederatedLearningMCP", "RobustMCP"), or "None" if no specific custom tool is directly applicable or if a generic configuration is more suitable.
- "mcp_config_name": The name of a pre-defined MCP configuration to use from mcp_configurations.json (e.g., "Standard HTTPS/JSON MCP"), or "None".
- "parameters": A dictionary of parameters needed for the MCP or to fulfill the query. For custom MCPs, these are parameters for their methods. If a parameter's value needs to be explicitly provided by the user later, indicate this (e.g., by setting its value to a placeholder like "USER_INPUT_REQUIRED" or by including it in the clarification question).
- "requires_further_clarification": boolean, true if you cannot determine the above with confidence or if essential parameters are missing.
- "clarification_question": string, a question to ask the user if 'requires_further_clarification' is true.
- "response_to_user": A direct, friendly response to the user if no tool is needed, if clarification is needed, or a simple acknowledgement.

Example User Query: "I need to process a stream of sensor data for anomalies."
Example JSON Output:
{{
  "intent": "Process real-time sensor data for anomaly detection",
  "mcp_tool_name": "RealTimeStreamingMCP",
  "mcp_config_name": null,
  "parameters": {{ "raw_data_json": "USER_INPUT_REQUIRED" }},
  "requires_further_clarification": true,
  "clarification_question": "Okay, I can use the RealTimeStreamingMCP for that. Please provide the sensor data stream as a JSON string.",
  "response_to_user": "Okay, I can use the RealTimeStreamingMCP for that. Please provide the sensor data stream as a JSON string."
}}

User Query: "{query}"

Your JSON Output:
"""
        
        llm_response_str = self._get_gemini_completion(
            prompt_text=prompt,
            max_output_tokens=1000, 
            temperature=0.1 # Low temperature for more predictable JSON
        )
        
        try:
            # Clean potential markdown ```json ... ```
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[7:]
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-3]
            llm_response_str = llm_response_str.strip()

            parsed_response = json.loads(llm_response_str)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: '{llm_response_str}'. Error: {e}")
            return {
                "intent": "Error in understanding query via LLM", 
                "mcp_tool_name": None, 
                "mcp_config_name": None,
                "parameters": {}, 
                "requires_further_clarification": True,
                "clarification_question": "I had some trouble interpreting that with Gemini. Could you please rephrase your request?",
                "response_to_user": "I'm having a little difficulty understanding your request with Gemini. Could you try phrasing it differently?"
            }

    def use_custom_mcp(self, mcp_name, method_name, params):
        """Invokes a method on a specified custom MCP instance."""
        # This method is identical to the one in previous agents.
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
        """Processes a user query, determines intent, uses tools, and manages memory."""
        # This method is largely identical to the one in previous agents.
        logging.info(f"\n--- Processing Query: '{query}' ---")

        intent_data = self._determine_intent_and_params_with_llm(query)
        response_to_user = intent_data.get("response_to_user", "I'm processing your request using Gemini.")
        
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

        if not mcp_tool_name and not mcp_config_name and not intent_data.get("requires_further_clarification") and response_to_user == "I'm processing your request using Gemini.":
            logging.info("No specific tool or config identified by LLM, and no direct response. Using general LLM completion.")
            fallback_prompt = f"User query: \"{query}\"\nProvide a helpful, general response:"
            response_to_user = self._get_gemini_completion(
                 prompt_text=fallback_prompt,
                 max_output_tokens=150 
            )
            self.add_procedural_memory("General query fallback", [query, f"LLM fallback response: {response_to_user}"])

        logging.info(f"Final Response: {response_to_user}")
        return response_to_user

# Example Usage
if __name__ == '__main__':
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-pro") # Or "gemini-1.5-flash-latest" etc.

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable must be set.")
        print("Skipping GeminiAgent demo.")
    else:
        print("--- GeminiAgent Demo ---")
        
        agent_specific_mcp_configs = {
            "streaming_mcp_config": {
                'validation_rules': { 'value': {'type': float, 'min': -10.0, 'max': 110.0}},
                'transformation_map': {} 
            },
            "federated_mcp_config": { "global_model_id": "gemini_agent_demo_model_v1" },
            "robust_mcp_config": { 'max_retries': 1, 'base_retry_delay_seconds': 0.1 }
        }

        try:
            agent = GeminiAgent(
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL_NAME,
                agent_config=agent_specific_mcp_configs
            )

            agent.add_factual_memory("user_city", "Mountain View")
            print(f"User's city: {agent.get_factual_memory('user_city')}")

            agent.add_procedural_memory("daily_report_generation", ["Fetch data", "Analyze trends", "Format report", "Send email"])
            print(f"Daily report steps: {agent.get_procedural_memory('daily_report_generation')}")

            queries = [
                "Hi, what are you capable of?",
                "I have a stream of data to process. It's: '[{\"sensor_id\": \"alpha1\", \"value\": 105.7}]'",
                "What was the last result from the RobustMCP?",
                "How do I send a message like '{\"id\": \"critical007\", \"payload\": {\"data\": \"payload_data\", \"force_failure_type\": \"permanent\"}}' in a robust way?",
                "Can you describe the 'IoT MQTT-based MCP' configuration?",
                "What city am I in, according to your records?" # Test factual memory via LLM
            ]

            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                # Gemini API might have rate limits.
                time.sleep(5) # Adjust as needed, especially if not using "gemini-pro" which can be slow.

            print("\n--- Final Memory States ---")
            print(f"Factual Memory: {json.dumps(agent.factual_memory, indent=2)}")
            print(f"Procedural Memory: {json.dumps(agent.procedural_memory, indent=2)}")

        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during the demo: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- End of GeminiAgent Demo ---")

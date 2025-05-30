import os
import json
import importlib.util
import time
import logging
from openai import AzureOpenAI # Keep existing client

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths (remain the same)
FACTUAL_MEMORY_FILE = "azure_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "azure_agent_procedural_memory.json"
# MCP Configuration File - THIS IS NEWLY CENTRAL
MCP_CONFIG_FILE_PATH = "mcp_configurations.json" # Agent will load this

class AzureOpenAIAgent:
    """
    An AI agent that utilizes Azure OpenAI services and is MCP-compliant,
    interacting with skills defined in mcp_configurations.json.
    """

    def __init__(self, azure_api_key, azure_endpoint, azure_api_version="2023-12-01-preview", llm_deployment_name="gpt-35-turbo", agent_config=None):
        if not all([azure_api_key, azure_endpoint, llm_deployment_name]):
            raise ValueError("Azure API key, endpoint, and LLM deployment name must be provided.")

        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.llm_deployment_name = llm_deployment_name
        self.agent_config = agent_config if agent_config else {} # General agent config, not MCP specific skill configs anymore directly here

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

        # Load MCP configurations
        self.mcp_registry = self._load_mcp_registry(MCP_CONFIG_FILE_PATH)
        if not self.mcp_registry:
            raise RuntimeError("Failed to load MCP configurations. Agent cannot operate.")

        logging.info("Agent initialized with memory and MCP registry.")

    def _load_memory(self, filepath):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f: memory = json.load(f)
                logging.info(f"Loaded memory from {filepath}.")
                return memory
            else:
                logging.info(f"Memory file {filepath} not found. Starting with empty memory.")
                return {}
        except Exception as e:
            logging.error(f"Error loading memory from {filepath}: {e}")
            return {}

    def _save_memory(self, filepath, data):
        try:
            with open(filepath, 'w') as f: json.dump(data, f, indent=4)
            logging.info(f"Saved memory to {filepath}.")
        except Exception as e: logging.error(f"Error saving memory to {filepath}: {e}")

    def add_factual_memory(self, fact_key, fact_value):
        self.factual_memory[fact_key] = fact_value
        self._save_memory(FACTUAL_MEMORY_FILE, self.factual_memory)
        logging.info(f"Added/Updated fact: '{fact_key}' = '{fact_value}'")

    def get_factual_memory(self, fact_key, default=None):
        return self.factual_memory.get(fact_key, default)

    def add_procedural_memory(self, task_name, steps):
        self.procedural_memory[task_name] = {"steps": steps, "last_used_timestamp": time.time()}
        self._save_memory(PROCEDURAL_MEMORY_FILE, self.procedural_memory)
        logging.info(f"Added/Updated procedure: '{task_name}'")

    def get_procedural_memory(self, task_name, default=None):
        return self.procedural_memory.get(task_name, default)

    def _load_mcp_registry(self, filepath):
        """Loads MCP configurations (skills, models, context_types)."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    registry = json.load(f)
                logging.info(f"Loaded MCP registry from {filepath}.")
                # Store skills in a dictionary for easy lookup by ID
                registry['skills_map'] = {skill['id']: skill for skill in registry.get('skills', [])}
                return registry
            else:
                logging.error(f"MCP configuration file {filepath} not found.")
                return None
        except Exception as e:
            logging.error(f"Error loading MCP registry from {filepath}: {e}")
            return None

    def _get_openai_chat_completion(self, messages, max_tokens=150, temperature=0.7):
        """Gets a chat completion from Azure OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logging.info("Azure OpenAI Chat API call successful.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Azure OpenAI Chat API call failed: {e}")
            if "DeploymentNotFound" in str(e) or "ResourceNotFound" in str(e):
                 return f"LLM_ERROR: Deployment '{self.llm_deployment_name}' not found." # Prefix for easier parsing
            return f"LLM_ERROR: Error communicating with Azure OpenAI (Chat): {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """Uses LLM to determine user intent, required skill_id, and parameters."""
        skills_info_for_prompt = [
            {"skill_id": skill['id'], "name": skill['name'], "description": skill['description'], "parameters": skill.get('parameters', [])}
            for skill in self.mcp_registry.get('skills', [])
        ]

        system_prompt = f"""
You are an AI assistant. Your task is to understand the user's query and determine if it can be addressed using one of your available skills.
Respond ONLY in JSON format with the following fields:
- "intent": A brief description of what the user wants to do.
- "skill_id": The ID of the skill to use (e.g., "realtime_data_processor", "federated_model_aggregator"). Choose from the list below. If no skill is appropriate, use "None".
- "parameters": A dictionary of parameters required by the chosen skill. These parameters must match the skill's definition.
- "requires_further_clarification": boolean, true if you cannot determine the above with confidence or if essential parameters for a skill are missing.
- "clarification_question": string, a question to ask the user if 'requires_further_clarification' is true.
- "response_to_user": A direct, friendly response to the user if no skill is needed or if clarification is needed.

Available Skills:
{json.dumps(skills_info_for_prompt, indent=2)}

Factual Memory Keys (for context, not for direct parameter filling unless user implies it): {list(self.factual_memory.keys())}

Example Query: "Process this sensor data: '[{{\"id\":\"s1\", \"val\":10}}]'"
Example JSON Response:
{{
  "intent": "Process real-time sensor data",
  "skill_id": "realtime_data_processor",
  "parameters": {{ "raw_data_json": "[{{\\"id\\":\\"s1\\", \\"val\\":10}}]" }},
  "requires_further_clarification": false,
  "clarification_question": "None",
  "response_to_user": "I will process your sensor data."
}}

User Query: "{query}"
Your JSON Output:
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        llm_response_str = self._get_openai_chat_completion(messages=messages, max_tokens=600, temperature=0.1)

        if llm_response_str.startswith("LLM_ERROR:"):
            logging.error(f"LLM communication error: {llm_response_str}")
            return {
                "intent": "LLM Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I'm having trouble with my core processing unit. Please try again later.",
                "response_to_user": "I'm currently unable to process requests. Please try again later."
            }
        try:
            parsed_response = json.loads(llm_response_str)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response as JSON: {llm_response_str}")
            return {
                "intent": "LLM Parsing Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I'm having trouble understanding the structure of that request. Could you rephrase?",
                "response_to_user": "My response parsing failed. Could you try asking in a different way?"
            }

    def _prepare_skill_context(self, skill_id: str) -> dict:
        """Prepares the context dictionary for a skill based on its definition."""
        context_data = {}
        skill_def = self.mcp_registry.get('skills_map', {}).get(skill_id)
        if not skill_def or 'expected_context' not in skill_def:
            return context_data # No specific context expected or skill not found

        for context_key_id in skill_def['expected_context']:
            context_type_def = next((ct for ct in self.mcp_registry.get('context_types', []) if ct['id'] == context_key_id), None)
            if not context_type_def:
                logging.warning(f"Definition for expected context key '{context_key_id}' not found in mcp_registry.")
                continue

            # This is where we'd populate context from agent's state/memory
            if context_key_id == "factual_memory_snapshot":
                context_data[context_key_id] = self.factual_memory.copy()
            elif context_key_id == "procedural_memory_snapshot":
                context_data[context_key_id] = self.procedural_memory.copy()
            # Add more context population logic as needed based on context_types
            # For now, skills primarily get data via parameters.
            else:
                # If skill expects other specific context data that agent might have, populate here.
                # Example: context_data[context_key_id] = self.some_internal_state.get(context_key_id_data)
                logging.info(f"Context key '{context_key_id}' expected by skill '{skill_id}', but no specific agent data source defined for it yet beyond memory snapshots.")

        # Add general agent config or other relevant info if skills might need it
        context_data['agent_name'] = "AzureOpenAIAgent"
        context_data['llm_deployment_name'] = self.llm_deployment_name

        logging.info(f"Prepared context for skill '{skill_id}': {list(context_data.keys())}")
        return context_data

    def invoke_mcp_skill(self, skill_id: str, parameters: dict, context_data: dict) -> dict:
        """Dynamically loads and invokes an MCP skill."""
        logging.info(f"Attempting to invoke skill '{skill_id}' with parameters: {parameters}")
        skill_def = self.mcp_registry.get('skills_map', {}).get(skill_id)

        if not skill_def:
            logging.error(f"Skill definition for ID '{skill_id}' not found in MCP registry.")
            return {"status": "error", "error_message": f"Skill '{skill_id}' not defined."}

        module_name = skill_def.get('handler_module')
        handler_path = skill_def.get('handler_class_or_function')

        if not module_name or not handler_path:
            logging.error(f"Skill '{skill_id}' is missing 'handler_module' or 'handler_class_or_function'.")
            return {"status": "error", "error_message": f"Skill '{skill_id}' is not configured correctly for invocation."}

        try:
            # Dynamically import the module
            # Assuming skill modules are in the same directory or Python path
            # Module path needs to be resolvable e.g. 'realtime_processing_skill'
            skill_module_path_py = f"{module_name}.py"

            # Check if module is already imported (e.g. by other agents or tests)
            if module_name in sys.modules:
                skill_module = sys.modules[module_name]
                # Optionally reload if changes are frequent during dev: importlib.reload(skill_module)
            else:
                spec = importlib.util.spec_from_file_location(module_name, skill_module_path_py)
                if spec is None or spec.loader is None: # Check spec and loader
                    raise ImportError(f"Could not create module spec for {module_name} at {skill_module_path_py}. Ensure file exists.")
                skill_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(skill_module)

            handler_parts = handler_path.split('.')
            obj_or_func = getattr(skill_module, handler_parts[0])

            if len(handler_parts) > 1: # Class method
                instance = obj_or_func() # Assumes constructor takes no args or uses skill_config from agent
                handler_method = getattr(instance, handler_parts[1])
                skill_response = handler_method(parameters, context_data)
            else: # Direct function
                skill_response = obj_or_func(parameters, context_data)

            logging.info(f"Skill '{skill_id}' executed successfully. Response status: {skill_response.get('status')}")
            return skill_response

        except FileNotFoundError:
            logging.error(f"Skill module file not found: {skill_module_path_py}")
            return {"status": "error", "error_message": f"Skill module '{module_name}.py' not found."}
        except (AttributeError, ImportError) as e:
            logging.error(f"Error loading/finding handler for skill '{skill_id}': {e}")
            return {"status": "error", "error_message": f"Could not load handler for skill '{skill_id}': {e}"}
        except Exception as e:
            logging.exception(f"Exception during skill '{skill_id}' execution: {e}") # Log full stack trace
            return {"status": "error", "error_message": f"Skill '{skill_id}' execution failed: {str(e)}"}

    def process_query(self, query: str) -> str:
        """Processes a user query using MCP-compliant skills."""
        logging.info(f"\n--- Processing Query (Azure Agent): '{query}' ---")

        intent_data = self._determine_intent_and_params_with_llm(query)

        skill_id = intent_data.get("skill_id")
        parameters = intent_data.get("parameters")
        response_to_user = intent_data.get("response_to_user", "I'm working on your request.")

        if intent_data.get("requires_further_clarification"):
            self.add_factual_memory("last_clarification_for_query", query)
            self.add_factual_memory("last_clarification_question", intent_data.get("clarification_question"))
            return intent_data.get("clarification_question", response_to_user)

        if skill_id and skill_id != "None":
            logging.info(f"LLM determined skill: '{skill_id}' with params: {parameters}")
            skill_context = self._prepare_skill_context(skill_id)
            skill_result = self.invoke_mcp_skill(skill_id, parameters, skill_context)

            if skill_result.get("status") == "success":
                # Agent might format this data further or use it to update memory/state
                # For now, construct a simple success message.
                response_to_user = f"Skill '{skill_id}' executed successfully. Output: {json.dumps(skill_result.get('data', 'No data returned'))}"
                self.add_procedural_memory(
                    task_name=f"Skill Execution: {skill_id}",
                    steps=[f"Query: {query}", f"Parameters: {parameters}", f"Result data: {skill_result.get('data')}"]
                )
                # Handle potential context updates suggested by the skill
                if "context_updates_suggestion" in skill_result:
                    logging.info(f"Skill '{skill_id}' suggested context updates: {skill_result['context_updates_suggestion']}")
                    # Here, the agent would decide how to merge/apply these suggestions to its main context or memory.
                    # For example, updating self.factual_memory or a session context.
                    # This part needs careful design based on how state is managed.
                    # E.g., self.factual_memory.update(skill_result['context_updates_suggestion'].get('factual_memory_snapshot', {}))

            else: # Error or other non-success status from skill
                error_msg = skill_result.get("error_message", "Skill execution failed with no specific error message.")
                response_to_user = f"Error executing skill '{skill_id}': {error_msg}"
                self.add_procedural_memory(
                    task_name=f"Skill Execution Failed: {skill_id}",
                    steps=[f"Query: {query}", f"Parameters: {parameters}", f"Error: {error_msg}"]
                )
        elif skill_id == "None" and intent_data.get("response_to_user"):
            # LLM decided no skill is needed and provided a direct response
            response_to_user = intent_data.get("response_to_user")
            self.add_procedural_memory("General query (no skill)", [query, f"LLM direct response: {response_to_user}"])
        else: # Fallback if LLM response was unclear or errored but didn't require clarification
            logging.warning("LLM did not identify a skill and no direct response was provided. Using general fallback.")
            response_to_user = "I understood your query, but I'm not sure how to act on it with my current skills."
            # Or, make another LLM call for a general chat response if desired.
            # For now, this simple message.

        logging.info(f"Final Response (Azure Agent): {response_to_user}")
        return response_to_user


if __name__ == '__main__':
    AZURE_OAI_KEY = os.environ.get("AZURE_OAI_KEY")
    AZURE_OAI_ENDPOINT = os.environ.get("AZURE_OAI_ENDPOINT")
    AZURE_OAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OAI_LLM_DEPLOYMENT", "gpt-35-turbo")

    if not AZURE_OAI_KEY or not AZURE_OAI_ENDPOINT:
        print("Error: AZURE_OAI_KEY and AZURE_OAI_ENDPOINT environment variables must be set.")
    elif not os.path.exists(MCP_CONFIG_FILE_PATH):
        print(f"Error: MCP Configuration file not found at {MCP_CONFIG_FILE_PATH}")
    elif not all(os.path.exists(f"{skill_def['handler_module']}.py") for skill_def in json.load(open(MCP_CONFIG_FILE_PATH))['skills']):
        print("Error: Not all skill handler module files (e.g., realtime_processing_skill.py) found in the current directory.")
    else:
        print("--- AzureOpenAIAgent MCP-Compliant Demo ---")
        try:
            agent = AzureOpenAIAgent(
                azure_api_key=AZURE_OAI_KEY,
                azure_endpoint=AZURE_OAI_ENDPOINT,
                llm_deployment_name=AZURE_OAI_LLM_DEPLOYMENT
            )

            agent.add_factual_memory("user_id", "user_123_azure")
            print(f"Initial factual memory: {agent.factual_memory}")

            queries = [
                "Hello there!",
                "Please process this data stream: '[{\"id\":\"sensor1\", \"temperature_celsius\":30}]'",
                "Start federated learning round 10.",
                "Submit an update for round 10 from participant P7 with data '{\"weights_delta\":[0.1], \"bias_delta\":0.01, \"data_samples_count\":50}'.",
                "Aggregate updates for round 10 using weighted_average.",
                "Handle this message robustly: '{\"id\":\"m001\", \"payload\":{\"data\":\"important info\"}}'",
                "What's the capital of France?" # Should be handled by LLM directly or general knowledge
            ]

            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                time.sleep(2)

            print("\n--- Final Factual Memory (Azure Agent) ---")
            print(json.dumps(agent.factual_memory, indent=2))
            print("\n--- Final Procedural Memory (Azure Agent) ---")
            print(json.dumps(agent.procedural_memory, indent=2))

        except RuntimeError as re:
            print(f"Runtime Error: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during the Azure agent demo: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- End of AzureOpenAIAgent MCP-Compliant Demo ---")

import os
import json
import importlib.util
import time
import logging
from anthropic import Anthropic # Anthropic library

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "anthropic_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "anthropic_agent_procedural_memory.json"
MCP_CONFIG_FILE_PATH = "mcp_configurations.json"

class AnthropicAgent:
    """
    An AI agent that utilizes Anthropic Claude services and is MCP-compliant,
    interacting with skills defined in mcp_configurations.json.
    """

    def __init__(self, api_key, model_name="claude-3-haiku-20240307", agent_config=None):
        if not api_key: raise ValueError("Anthropic API key must be provided.")
        if not model_name: raise ValueError("Anthropic model name must be provided.")

        self.api_key = api_key
        self.model_name = model_name
        self.agent_config = agent_config if agent_config else {}

        try:
            self.anthropic_client = Anthropic(api_key=self.api_key)
            logging.info(f"Anthropic client initialized successfully for model {self.model_name}.")
        except Exception as e:
            logging.error(f"Failed to initialize Anthropic client: {e}")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)

        self.mcp_registry = self._load_mcp_registry(MCP_CONFIG_FILE_PATH)
        if not self.mcp_registry:
            raise RuntimeError("Failed to load MCP configurations. Agent cannot operate.")

        logging.info("Anthropic Agent initialized with memory and MCP registry.")

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

    def get_factual_memory(self, fact_key, default=None):
        return self.factual_memory.get(fact_key, default)

    def add_procedural_memory(self, task_name, steps):
        self.procedural_memory[task_name] = {"steps": steps, "last_used_timestamp": time.time()}
        self._save_memory(PROCEDURAL_MEMORY_FILE, self.procedural_memory)

    def get_procedural_memory(self, task_name, default=None):
        return self.procedural_memory.get(task_name, default)

    def _load_mcp_registry(self, filepath):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f: registry = json.load(f)
                logging.info(f"Loaded MCP registry from {filepath}.")
                registry['skills_map'] = {skill['id']: skill for skill in registry.get('skills', [])}
                return registry
            else:
                logging.error(f"MCP configuration file {filepath} not found.")
                return None
        except Exception as e:
            logging.error(f"Error loading MCP registry from {filepath}: {e}")
            return None

    def _get_anthropic_completion(self, system_prompt_text, user_prompt_text, max_tokens=1024, temperature=0.1):
        """Gets a completion from Anthropic using the Messages API."""
        try:
            message = self.anthropic_client.messages.create(
                model=self.model_name,
                system=system_prompt_text,
                messages=[{"role": "user", "content": user_prompt_text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            logging.info(f"Anthropic API call successful (Model: {self.model_name}).")
            if message.content and isinstance(message.content, list) and hasattr(message.content[0], 'text'):
                return message.content[0].text
            else:
                logging.error("Anthropic API response format not as expected or empty.")
                return "LLM_ERROR: Received unexpected response format from Anthropic."
        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            return f"LLM_ERROR: Error communicating with Anthropic: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """Uses LLM to determine user intent, required skill_id, and parameters."""
        skills_info_for_prompt = [
            {"skill_id": skill['id'], "name": skill['name'], "description": skill['description'], "parameters": skill.get('parameters', [])}
            for skill in self.mcp_registry.get('skills', [])
        ]

        # Anthropic models respond well to XML-like structures for complex instructions.
        system_prompt = f"""
You are an AI assistant. Your task is to understand a user's query and determine if it can be addressed using one of your available skills.
You must respond ONLY in a structured JSON format. Do not include any other text before or after the JSON.

Here are the available skills:
<skills_list>
{json.dumps(skills_info_for_prompt, indent=2)}
</skills_list>

Here are keys from your factual memory (for context only):
<factual_memory_keys>
{list(self.factual_memory.keys())}
</factual_memory_keys>

Based on the user's query, output a JSON object with the following fields:
- "intent": A brief description of what the user wants to do.
- "skill_id": The ID of the skill to use (e.g., "realtime_data_processor"). Choose from the list above. If no skill is appropriate, use "None".
- "parameters": A dictionary of parameters required by the chosen skill, matching the skill's definition.
- "requires_further_clarification": boolean, true if essential parameters are missing or intent is unclear.
- "clarification_question": string, a question for the user if clarification is needed.
- "response_to_user": A direct, friendly response if no skill is needed or for clarification.

Example User Query: "Process this sensor data: '[{{\"id\":\"s1\", \"val\":10}}]'"
Example JSON Output:
{{
  "intent": "Process real-time sensor data",
  "skill_id": "realtime_data_processor",
  "parameters": {{ "raw_data_json": "[{{\\"id\\":\\"s1\\", \\"val\\":10}}]" }},
  "requires_further_clarification": false,
  "clarification_question": "None",
  "response_to_user": "I will process your sensor data."
}}
"""
        # The user query will be the "user" message content.
        user_prompt_for_llm = f"User Query: \"{query}\"\nYour JSON Output:"


        llm_response_str = self._get_anthropic_completion(
            system_prompt_text=system_prompt,
            user_prompt_text=user_prompt_for_llm, # Query is passed as user message
            max_tokens=1000, # Ensure enough for JSON
            temperature=0.1
        )

        if llm_response_str.startswith("LLM_ERROR:"):
            logging.error(f"LLM communication error: {llm_response_str}")
            return {
                "intent": "LLM Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I'm having trouble with my connection to the language model (Anthropic). Please try again later.",
                "response_to_user": "I'm currently unable to process requests due to an LLM connection issue with Anthropic. Please try again later."
            }
        try:
            # Attempt to extract JSON, as Anthropic might sometimes add preamble/postamble
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                llm_response_str_cleaned = llm_response_str[json_start:json_end]
            else:
                llm_response_str_cleaned = llm_response_str # Fallback if no clear JSON object delimiters

            parsed_response = json.loads(llm_response_str_cleaned)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response as JSON: {llm_response_str}")
            return {
                "intent": "LLM Parsing Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I received an unexpected response format from the language model (Anthropic). Could you rephrase?",
                "response_to_user": "My response parsing failed (Anthropic). Could you try asking in a different way?"
            }

    def _prepare_skill_context(self, skill_id: str) -> dict:
        """Prepares the context dictionary for a skill."""
        context_data = {}
        skill_def = self.mcp_registry.get('skills_map', {}).get(skill_id)
        if not skill_def or 'expected_context' not in skill_def:
            return context_data

        for context_key_id in skill_def['expected_context']:
            if context_key_id == "factual_memory_snapshot":
                context_data[context_key_id] = self.factual_memory.copy()
            elif context_key_id == "procedural_memory_snapshot":
                context_data[context_key_id] = self.procedural_memory.copy()
            else:
                logging.info(f"Context key '{context_key_id}' expected by skill '{skill_id}', no specific data source defined yet.")

        context_data['agent_name'] = "AnthropicAgent"
        context_data['llm_model_name'] = self.model_name
        return context_data

    def invoke_mcp_skill(self, skill_id: str, parameters: dict, context_data: dict) -> dict:
        """Dynamically loads and invokes an MCP skill."""
        logging.info(f"Attempting to invoke skill '{skill_id}' with parameters: {parameters}")
        skill_def = self.mcp_registry.get('skills_map', {}).get(skill_id)

        if not skill_def:
            return {"status": "error", "error_message": f"Skill '{skill_id}' not defined."}

        module_name = skill_def.get('handler_module')
        handler_path = skill_def.get('handler_class_or_function')

        if not module_name or not handler_path:
            return {"status": "error", "error_message": f"Skill '{skill_id}' is not configured correctly for invocation."}
        try:
            skill_module_path_py = f"{module_name}.py"
            if module_name in sys.modules:
                skill_module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, skill_module_path_py)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {module_name} at {skill_module_path_py}.")
                skill_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(skill_module)

            handler_parts = handler_path.split('.')
            obj_or_func = getattr(skill_module, handler_parts[0])

            if len(handler_parts) > 1: # Class method
                instance = obj_or_func()
                handler_method = getattr(instance, handler_parts[1])
                skill_response = handler_method(parameters, context_data)
            else: # Direct function
                skill_response = obj_or_func(parameters, context_data)

            logging.info(f"Skill '{skill_id}' executed. Response status: {skill_response.get('status')}")
            return skill_response
        except FileNotFoundError:
            logging.error(f"Skill module file not found: {skill_module_path_py}")
            return {"status": "error", "error_message": f"Skill module '{module_name}.py' not found."}
        except (AttributeError, ImportError) as e:
            return {"status": "error", "error_message": f"Could not load handler for skill '{skill_id}': {e}"}
        except Exception as e:
            logging.exception(f"Exception during skill '{skill_id}' execution: {e}")
            return {"status": "error", "error_message": f"Skill '{skill_id}' execution failed: {str(e)}"}

    def process_query(self, query: str) -> str:
        logging.info(f"\n--- Processing Query (Anthropic Agent): '{query}' ---")
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
                response_to_user = f"Skill '{skill_id}' executed successfully. Output: {json.dumps(skill_result.get('data', 'No data returned'))}"
                self.add_procedural_memory(f"Skill Execution: {skill_id}", [f"Query: {query}", f"Params: {parameters}", f"Result: {skill_result.get('data')}"])
                if "context_updates_suggestion" in skill_result:
                    logging.info(f"Skill '{skill_id}' suggested context updates: {skill_result['context_updates_suggestion']}")
            else:
                error_msg = skill_result.get("error_message", "Skill execution failed.")
                response_to_user = f"Error executing skill '{skill_id}': {error_msg}"
                self.add_procedural_memory(f"Skill Execution Failed: {skill_id}", [f"Query: {query}", f"Params: {parameters}", f"Error: {error_msg}"])
        elif skill_id == "None" and intent_data.get("response_to_user"):
            response_to_user = intent_data.get("response_to_user")
            self.add_procedural_memory("General query (no skill)", [query, f"LLM direct response: {response_to_user}"])
        else:
            logging.warning("LLM did not identify a skill and no direct response. Using general fallback.")
            response_to_user = "I understood your query, but I'm not sure how to act on it with my current skills (Anthropic)."
            # Fallback to a general LLM call if no skill and no direct response
            # system_fallback_prompt = "You are a helpful AI assistant. Respond to the user's query."
            # response_to_user = self._get_anthropic_completion(system_fallback_prompt, query, max_tokens=150)
            # self.add_procedural_memory("General query (fallback)", [query, f"LLM fallback response: {response_to_user}"])


        logging.info(f"Final Response (Anthropic Agent): {response_to_user}")
        return response_to_user

if __name__ == '__main__':
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.environ.get("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable must be set.")
    elif not os.path.exists(MCP_CONFIG_FILE_PATH):
        print(f"Error: MCP Configuration file not found at {MCP_CONFIG_FILE_PATH}")
    elif not all(os.path.exists(f"{skill_def['handler_module']}.py") for skill_def in json.load(open(MCP_CONFIG_FILE_PATH))['skills']):
        print("Error: Not all skill handler module files found in the current directory.")
    else:
        print("--- AnthropicAgent MCP-Compliant Demo ---")
        try:
            agent = AnthropicAgent(
                api_key=ANTHROPIC_API_KEY,
                model_name=ANTHROPIC_MODEL_NAME
            )
            agent.add_factual_memory("user_project_id", "project_claude_x")
            print(f"Initial factual memory: {agent.factual_memory}")

            queries = [
                "What is your purpose?",
                "I have sensor data: '[{\"id\":\"s7\", \"temperature_celsius\":22.5}]'. Can you process it?",
                "I need to start round 3 for federated learning.",
                "What is the current global model for federated learning?",
                "Process this message with retries: '{\"id\":\"m77\", \"payload\":{\"info\":\"handle with care\"}}'"
            ]
            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                time.sleep(5) # Anthropic API might have rate limits

            print("\n--- Final Factual Memory (Anthropic Agent) ---")
            print(json.dumps(agent.factual_memory, indent=2))
            print("\n--- Final Procedural Memory (Anthropic Agent) ---")
            print(json.dumps(agent.procedural_memory, indent=2))

        except RuntimeError as re:
            print(f"Runtime Error: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during the Anthropic agent demo: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- End of AnthropicAgent MCP-Compliant Demo ---")

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
MCP_CONFIG_FILE_PATH = "mcp_configurations.json"

class GeminiAgent:
    """
    An AI agent that utilizes Google Gemini services and is MCP-compliant,
    interacting with skills defined in mcp_configurations.json.
    """

    def __init__(self, api_key, model_name="gemini-pro", agent_config=None):
        if not api_key: raise ValueError("Google Gemini API key must be provided.")
        if not model_name: raise ValueError("Google Gemini model name must be provided.")

        self.api_key = api_key
        self.model_name = model_name
        self.agent_config = agent_config if agent_config else {}

        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(model_name=self.model_name)
            logging.info(f"Google Gemini client initialized successfully for model {self.model_name}.")
        except Exception as e:
            logging.error(f"Failed to initialize Google Gemini client: {e}")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)

        self.mcp_registry = self._load_mcp_registry(MCP_CONFIG_FILE_PATH)
        if not self.mcp_registry:
            raise RuntimeError("Failed to load MCP configurations. Agent cannot operate.")

        logging.info("Gemini Agent initialized with memory and MCP registry.")

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

    def _get_gemini_completion(self, prompt_text, max_output_tokens=1024, temperature=0.1):
        """Gets a completion from Google Gemini."""
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
            response = self.gemini_model.generate_content(
                prompt_text,
                generation_config=generation_config
            )
            logging.info(f"Google Gemini API call successful (Model: {self.model_name}).")

            if not response.candidates or response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.prompt_feedback.safety_ratings]) if response.prompt_feedback.safety_ratings else "N/A"
                logging.error(f"Gemini API call blocked. Reason: {block_reason}. Safety: [{safety_ratings_str}]")
                return f"LLM_ERROR: Gemini API call blocked due to '{block_reason}'. Safety ratings: [{safety_ratings_str}]"

            if response.text: return response.text
            elif response.parts: return "".join(part.text for part in response.parts if hasattr(part, "text"))
            else:
                logging.error("Gemini API response format not as expected or empty content.")
                return "LLM_ERROR: Received unexpected response format or empty content from Gemini."
        except Exception as e:
            logging.error(f"Google Gemini API call failed: {e}")
            return f"LLM_ERROR: Error communicating with Google Gemini: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """Uses LLM to determine user intent, required skill_id, and parameters."""
        skills_info_for_prompt = [
            {"skill_id": skill['id'], "name": skill['name'], "description": skill['description'], "parameters": skill.get('parameters', [])}
            for skill in self.mcp_registry.get('skills', [])
        ]

        # Gemini prompt structure
        prompt = f"""
You are an AI assistant. Your task is to understand the user's query and determine if it can be addressed using one of your available skills.
Respond ONLY in a structured JSON format. Do not include any markdown specifiers like ```json ... ``` or any text before or after the JSON object.

Available Skills:
{json.dumps(skills_info_for_prompt, indent=2)}

Factual Memory Keys (for context only): {list(self.factual_memory.keys())}

Based on the user's query, output a JSON object with the following fields:
- "intent": A brief description of what the user wants to do.
- "skill_id": The ID of the skill to use (e.g., "realtime_data_processor"). Choose from the list above. If no skill is appropriate, use "None".
- "parameters": A dictionary of parameters required by the chosen skill, matching the skill's definition.
- "requires_further_clarification": boolean, true if essential parameters are missing or intent is unclear.
- "clarification_question": string, a question for the user if clarification is needed.
- "response_to_user": A direct, friendly response if no skill is needed or for clarification.

User Query: "{query}"
Your JSON Output:
"""
        llm_response_str = self._get_gemini_completion(prompt_text=prompt, max_output_tokens=1000, temperature=0.1)

        if llm_response_str.startswith("LLM_ERROR:"):
            logging.error(f"LLM communication error: {llm_response_str}")
            return {
                "intent": "LLM Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I'm having trouble with my connection to the Gemini language model. Please try again later.",
                "response_to_user": "I'm currently unable to process requests due to an LLM connection issue with Gemini. Please try again later."
            }
        try:
            # Clean potential markdown (though instructed Gemini not to use it)
            if llm_response_str.startswith("```json"): llm_response_str = llm_response_str[7:]
            if llm_response_str.endswith("```"): llm_response_str = llm_response_str[:-3]
            llm_response_str = llm_response_str.strip()

            parsed_response = json.loads(llm_response_str)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response as JSON: {llm_response_str}")
            return {
                "intent": "LLM Parsing Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I received an unexpected response format from the Gemini model. Could you rephrase?",
                "response_to_user": "My response parsing failed (Gemini). Could you try asking in a different way?"
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

        context_data['agent_name'] = "GeminiAgent"
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
        logging.info(f"\n--- Processing Query (Gemini Agent): '{query}' ---")
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
            response_to_user = "I understood your query, but I'm not sure how to act on it with my current skills (Gemini)."
            # Fallback_prompt for Gemini if needed
            # fallback_prompt = f"User query: \"{query}\"\nProvide a helpful, general response:"
            # response_to_user = self._get_gemini_completion(prompt_text=fallback_prompt, max_output_tokens=150)
            # self.add_procedural_memory("General query (fallback)", [query, f"LLM fallback response: {response_to_user}"])

        logging.info(f"Final Response (Gemini Agent): {response_to_user}")
        return response_to_user

if __name__ == '__main__':
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-pro")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable must be set.")
    elif not os.path.exists(MCP_CONFIG_FILE_PATH):
        print(f"Error: MCP Configuration file not found at {MCP_CONFIG_FILE_PATH}")
    elif not all(os.path.exists(f"{skill_def['handler_module']}.py") for skill_def in json.load(open(MCP_CONFIG_FILE_PATH))['skills']):
        print("Error: Not all skill handler module files found in the current directory.")
    else:
        print("--- GeminiAgent MCP-Compliant Demo ---")
        try:
            agent = GeminiAgent(
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL_NAME
            )
            agent.add_factual_memory("user_location_preference", "online")
            print(f"Initial factual memory: {agent.factual_memory}")

            queries = [
                "How are you today?",
                "I have a data stream: '[{\"id\":\"t1\", \"temperature_celsius\":10.0}]', please process it.",
                "For our federated learning task, model 'alpha', start round 1.", # Model ID is part of params for FL skill if designed so
                "Send this message with robust handling: '{\"id\":\"err_test\", \"payload\":{\"data\":\"test_permanent_error\", \"force_failure_type\":\"permanent\"}}'"
            ]
            for q in queries:
                response = agent.process_query(q)
                print(f"\nUser Query: {q}\nAgent Response: {response}")
                time.sleep(5) # Gemini API can have strict rate limits

            print("\n--- Final Factual Memory (Gemini Agent) ---")
            print(json.dumps(agent.factual_memory, indent=2))
            print("\n--- Final Procedural Memory (Gemini Agent) ---")
            print(json.dumps(agent.procedural_memory, indent=2))

        except RuntimeError as re:
            print(f"Runtime Error: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during the Gemini agent demo: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- End of GeminiAgent MCP-Compliant Demo ---")

import os
import json
import importlib.util
import time
import logging
from transformers import pipeline # Hugging Face library

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "huggingface_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "huggingface_agent_procedural_memory.json"
MCP_CONFIG_FILE_PATH = "mcp_configurations.json"

class HuggingFaceAgent:
    """
    An AI agent that utilizes a local Hugging Face Transformer model and is MCP-compliant,
    interacting with skills defined in mcp_configurations.json.
    """

    def __init__(self, model_name="gpt2", task="text-generation", agent_config=None, device=-1):
        if not model_name: raise ValueError("Hugging Face model name must be provided.")

        self.model_name = model_name
        self.task = task
        self.agent_config = agent_config if agent_config else {}

        try:
            logging.info(f"Loading Hugging Face model: {self.model_name} for task: {self.task}. This may take time...")
            self.hf_pipeline = pipeline(self.task, model=self.model_name, tokenizer=self.model_name, device=device)
            # Ensure pad_token_id is set for models like GPT-2 if not already set by pipeline
            if self.hf_pipeline.tokenizer.pad_token_id is None and self.hf_pipeline.model.config.eos_token_id is not None:
                self.hf_pipeline.tokenizer.pad_token_id = self.hf_pipeline.model.config.eos_token_id
            logging.info(f"Hugging Face pipeline initialized for model {self.model_name} on device {device}.")
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face pipeline for model {self.model_name}: {e}")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)

        self.mcp_registry = self._load_mcp_registry(MCP_CONFIG_FILE_PATH)
        if not self.mcp_registry:
            raise RuntimeError("Failed to load MCP configurations. Agent cannot operate.")

        logging.info("HuggingFaceAgent initialized with memory and MCP registry.")

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

    def _get_huggingface_completion(self, prompt_text, max_new_tokens=250, temperature=0.7, top_p=0.9):
        """Gets a completion from the loaded Hugging Face model."""
        try:
            # Ensure pad_token_id is set, especially for open-ended text generation.
            # This might have been set in __init__ but good to double check or ensure it's handled.
            if self.hf_pipeline.tokenizer.pad_token_id is None:
                 self.hf_pipeline.tokenizer.pad_token_id = self.hf_pipeline.model.config.eos_token_id

            sequences = self.hf_pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature if temperature > 0 else None, # Temp must be positive for sampling
                top_p=top_p if temperature > 0 else None, # top_p only if sampling
                do_sample=True if temperature > 0 else False, # Enable sampling if temp is set
                pad_token_id=self.hf_pipeline.tokenizer.pad_token_id
            )
            logging.info(f"Hugging Face model '{self.model_name}' API call successful.")
            generated_text = sequences[0]['generated_text']
            if generated_text.startswith(prompt_text):
                return generated_text[len(prompt_text):].strip()
            return generated_text.strip()
        except Exception as e:
            logging.error(f"Hugging Face model '{self.model_name}' call failed: {e}")
            return f"LLM_ERROR: Error with Hugging Face model: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """Uses local Hugging Face model for intent, skill_id, and parameters."""
        skills_info_for_prompt = [
            {"skill_id": skill['id'], "name": skill['name'], "description": skill['description'], "parameters": skill.get('parameters', [])}
            for skill in self.mcp_registry.get('skills', [])
        ]

        prompt = f"""
Instruction: You are an AI assistant. Analyze the user's query and available tools to determine the user's intent and how to fulfill it.
You have access to skills. Respond ONLY with a JSON object matching the specified structure. Do not add any explanations or markdown.

Available Skills:
{json.dumps(skills_info_for_prompt, indent=2)}

Factual Memory Keys (for context): {list(self.factual_memory.keys())}

JSON Output Structure:
{{
  "intent": "Description of user's goal.",
  "skill_id": "ID of the skill or 'None'.",
  "parameters": {{ "param1": "value1", ... }},
  "requires_further_clarification": true/false,
  "clarification_question": "Question if clarification is needed, else 'None'.",
  "response_to_user": "A direct, friendly response to the user."
}}

Example User Query: "Process sensor data: '[{{\"id\":\"s1\", \"val\":10}}]'"
Example JSON Output:
{{
  "intent": "Process real-time sensor data",
  "skill_id": "realtime_data_processor",
  "parameters": {{ "raw_data_json": "[{{\\"id\\":\\"s1\\", \\"val\\":10}}]" }},
  "requires_further_clarification": false,
  "clarification_question": "None",
  "response_to_user": "I will process your sensor data."
}}

User Query: "{query}"
JSON Output:
"""
        # Using low temperature for more deterministic JSON, but sampling still on.
        llm_response_str = self._get_huggingface_completion(prompt, max_new_tokens=400, temperature=0.1, top_p=0.7)

        if llm_response_str.startswith("LLM_ERROR:"):
            logging.error(f"LLM communication error: {llm_response_str}")
            return {
                "intent": "LLM Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "I'm having trouble with my local language model. Please try again later.",
                "response_to_user": "I'm currently unable to process requests due to an issue with my local model. Please try again later."
            }
        try:
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                llm_response_str_cleaned = llm_response_str[json_start:json_end]
            else: # If no clear JSON block, assume it's a malformed attempt
                raise json.JSONDecodeError("No valid JSON block found in LLM response.", llm_response_str, 0)

            parsed_response = json.loads(llm_response_str_cleaned)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: '{llm_response_str}'. Error: {e}")
            return {
                "intent": "LLM Parsing Error", "skill_id": None, "parameters": {},
                "requires_further_clarification": True,
                "clarification_question": "My local model provided a response I couldn't parse. Could you simplify or rephrase?",
                "response_to_user": f"The local model's response was not clear (raw: '{llm_response_str[:100]}...'). Try rephrasing."
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

        context_data['agent_name'] = "HuggingFaceAgent"
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
        logging.info(f"\n--- Processing Query (HuggingFace Agent): '{query}' ---")
        intent_data = self._determine_intent_and_params_with_llm(query)

        skill_id = intent_data.get("skill_id")
        parameters = intent_data.get("parameters")
        response_to_user = intent_data.get("response_to_user", "I'm working on your request with my local model.")

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
            response_to_user = "I understood your query, but I'm not sure how to act on it with my current skills (Local Model)."
            # Fallback prompt for HF model
            # fallback_prompt = f"User query: \"{query}\"\nProvide a helpful, general response:"
            # response_to_user = self._get_huggingface_completion(fallback_prompt, max_new_tokens=100, temperature=0.5)
            # self.add_procedural_memory("General query (fallback)", [query, f"LLM fallback response: {response_to_user}"])

        logging.info(f"Final Response (HuggingFace Agent): {response_to_user}")
        return response_to_user

if __name__ == '__main__':
    HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "gpt2")
    HF_DEVICE = int(os.environ.get("HF_DEVICE", -1))

    if not os.path.exists(MCP_CONFIG_FILE_PATH):
        print(f"Error: MCP Configuration file not found at {MCP_CONFIG_FILE_PATH}")
    elif not all(os.path.exists(f"{skill_def['handler_module']}.py") for skill_def in json.load(open(MCP_CONFIG_FILE_PATH))['skills']):
        print("Error: Not all skill handler module files found in the current directory.")
    else:
        print(f"--- HuggingFaceAgent MCP-Compliant Demo (Model: {HF_MODEL_NAME}) ---")
        try:
            agent = HuggingFaceAgent(
                model_name=HF_MODEL_NAME,
                device=HF_DEVICE
            )
            agent.add_factual_memory("user_setting_theme", "dark_mode")
            print(f"Initial factual memory: {agent.factual_memory}")

            queries = [
                "What can you do for me?",
                "I have data: '[{\"id\":\"local_sensor\", \"temperature_celsius\":19.5}]'. Use skill realtime_data_processor.", # More direct for local model
                "For federated learning, I want to start round 1 using action 'start_new_round'.", # More direct
                "I want to use skill 'robust_message_handler' for message '{\"id\":\"local_m1\", \"payload\":{\"detail\":\"test\"}}'." # More direct
            ]
            for q_idx, q in enumerate(queries):
                print(f"\n--- Query {q_idx+1} ---")
                response = agent.process_query(q)
                print(f"User Query: {q}\nAgent Response: {response}")
                time.sleep(1)

            print("\n--- Final Factual Memory (HuggingFace Agent) ---")
            print(json.dumps(agent.factual_memory, indent=2))
            print("\n--- Final Procedural Memory (HuggingFace Agent) ---")
            print(json.dumps(agent.procedural_memory, indent=2))

        except RuntimeError as re:
            print(f"Runtime Error: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during the HuggingFace agent demo: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- End of HuggingFaceAgent MCP-Compliant Demo ---")

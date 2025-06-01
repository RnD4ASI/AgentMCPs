import os
import json
import importlib.util
import time
import logging
import uuid # Added for A2A agent_id
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # Hugging Face library
from a2a_protocol import AgentCard, A2ATask, A2AClient # A2A imports

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memory file paths
FACTUAL_MEMORY_FILE = "huggingface_agent_factual_memory.json"
PROCEDURAL_MEMORY_FILE = "huggingface_agent_procedural_memory.json"
MCP_CONFIG_FILE = "mcp_configurations.json"

# Custom MCP module paths
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

class HuggingFaceAgent:
    """
    An AI agent that utilizes a local Hugging Face Transformer model and custom MCP tools,
    with capabilities for managing factual and procedural memory.
    """

    def __init__(self, model_name="gpt2", task="text-generation", agent_config=None, device=-1): # device: -1 for CPU, 0 for GPU if available
        """
        Initializes the HuggingFaceAgent.

        Args:
            model_name (str): Name of the Hugging Face model to load (e.g., "gpt2", "distilgpt2", "google/flan-t5-base").
            task (str): The task for the Hugging Face pipeline (e.g., "text-generation", "summarization").
            agent_config (dict, optional): Configuration for the agent, including MCP configs.
            device (int): Device to run the model on (-1 for CPU, 0 for GPU).
        """
        if not model_name:
            raise ValueError("Hugging Face model name must be provided.")

        self.model_name = model_name
        self.task = task
        self.agent_config = agent_config if agent_config else {}

        try:
            # For text-generation, especially for structured output, direct model and tokenizer usage
            # can sometimes offer more control than pipeline, but pipeline is simpler to start.
            # Using pipeline for now.
            logging.info(f"Loading Hugging Face model: {self.model_name} for task: {self.task}. This may take some time...")
            # For tasks like text-generation expecting specific JSON, more control might be needed.
            # However, for simplicity, using pipeline first.
            # Some models might need specific trust_remote_code=True, but use with caution.
            self.hf_pipeline = pipeline(self.task, model=self.model_name, tokenizer=self.model_name, device=device)
            logging.info(f"Hugging Face pipeline initialized successfully for model {self.model_name} on device {device}.")
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face pipeline for model {self.model_name}: {e}")
            logging.warning("Ensure the model name is correct and you have an internet connection if the model isn't cached.")
            logging.warning("Some models might require `trust_remote_code=True` if they have custom code, use with caution.")
            raise

        self.factual_memory = self._load_memory(FACTUAL_MEMORY_FILE)
        self.procedural_memory = self._load_memory(PROCEDURAL_MEMORY_FILE)
        
        self.mcp_configurations = self._load_mcp_configs(MCP_CONFIG_FILE)
        self.custom_mcps = self._load_custom_mcps()

        # A2A Initialization
        self.a2a_client = A2AClient()
        self.agent_id = f"huggingface-agent-{str(uuid.uuid4())}"
        self.a2a_endpoint_url = f"http://localhost:8005/a2a/{self.agent_id}" # Example endpoint
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
        self.factual_memory[fact_key] = fact_value
        self._save_memory(FACTUAL_MEMORY_FILE, self.factual_memory)
        logging.info(f"Added/Updated fact: '{fact_key}' = '{fact_value}'")

    def get_factual_memory(self, fact_key, default=None):
        value = self.factual_memory.get(fact_key, default)
        logging.info(f"Retrieved fact: '{fact_key}' = '{value}'")
        return value

    def add_procedural_memory(self, task_name, steps):
        self.procedural_memory[task_name] = {
            "steps": steps,
            "last_used_timestamp": time.time()
        }
        self._save_memory(PROCEDURAL_MEMORY_FILE, self.procedural_memory)
        logging.info(f"Added/Updated procedure: '{task_name}' with {len(steps)} steps.")

    def get_procedural_memory(self, task_name, default=None):
        procedure = self.procedural_memory.get(task_name, default)
        if procedure: logging.info(f"Retrieved procedure: '{task_name}'")
        else: logging.info(f"Procedure '{task_name}' not found.")
        return procedure
        
    def find_relevant_procedure(self, query_keywords):
        for task_name, procedure_data in self.procedural_memory.items():
            if any(keyword.lower() in task_name.lower() for keyword in query_keywords):
                logging.info(f"Found relevant procedure '{task_name}' for keywords: {query_keywords}")
                return procedure_data
        logging.info(f"No specific procedure found for keywords: {query_keywords}")
        return None

    def _load_mcp_configs(self, filepath):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f: configs = json.load(f)
                logging.info(f"Loaded MCP configurations from {filepath}.")
                return configs
            else:
                logging.warning(f"MCP configuration file {filepath} not found.")
                return []
        except Exception as e:
            logging.error(f"Error loading MCP configurations from {filepath}: {e}")
            return []

    def _load_custom_mcps(self):
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
                else: logging.error(f"Could not create spec for module {mcp_details['module_name']} at {module_path}")
            except FileNotFoundError: logging.error(f"Custom MCP file {module_path} not found.")
            except AttributeError: logging.error(f"Class {mcp_details['class_name']} not found in module {mcp_details['module_name']}.")
            except Exception as e: logging.error(f"Failed to load custom MCP {mcp_name}: {e}")
        return mcps

    def _get_huggingface_completion(self, prompt_text, max_length=250, num_return_sequences=1, temperature=0.7, top_p=0.9):
        """
        Gets a completion from the loaded Hugging Face model.
        Adjust max_length based on the expected JSON output size.
        """
        try:
            # For text-generation, the input is the prompt itself.
            # The pipeline handles tokenization and decoding.
            # Ensure prompt_text is not excessively long for the model's context window.
            # Max length should accommodate prompt + generated JSON.
            # For smaller models like GPT-2, max_length is often 1024 total (prompt + completion)
            # We set max_new_tokens to control generation length specifically.
            
            # The `max_length` parameter in pipeline often means total length (prompt + new tokens)
            # `max_new_tokens` is usually preferred for more direct control over output length.
            # Check your specific model/pipeline documentation.
            # For GPT-2 pipeline, max_length is a common parameter.
            
            # Truncate prompt if too long (very basic truncation)
            # A better approach would be to use tokenizer to check length.
            # Max model length for gpt2 is 1024 tokens.
            # Let's assume prompt is not excessively long for this example.
            
            # Some pipelines might use 'max_new_tokens' instead of 'max_length' for the generated part.
            # The 'text-generation' pipeline with GPT-2 usually takes 'max_length' for total length.
            # Let's try to be more explicit if possible or rely on pipeline defaults.
            # Forcing generation of a certain number of new tokens:
            # A bit of a guess for max_new_tokens, should be enough for the JSON.
            # The actual parameter name might vary slightly based on underlying model type / pipeline implementation.
            # Common ones: max_new_tokens, max_length (for total length).
            # If max_length is used, it must be > len(prompt_tokens).
            
            # Using a simple approach for now.
            # Pad token ID is important for some models if generating shorter sequences than max_length
            if self.hf_pipeline.tokenizer.pad_token_id is None:
                 self.hf_pipeline.tokenizer.pad_token_id = self.hf_pipeline.model.config.eos_token_id

            sequences = self.hf_pipeline(
                prompt_text,
                max_new_tokens=max_length, # Max tokens to generate *after* the prompt
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True, # Necessary for temperature and top_p to have an effect
                pad_token_id=self.hf_pipeline.tokenizer.eos_token_id # Suppress warning
            )
            
            logging.info(f"Hugging Face model '{self.model_name}' API call successful.")
            # The output is a list of dicts, each with 'generated_text'.
            # We need the text generated *after* the prompt.
            generated_text = sequences[0]['generated_text']
            # Remove the prompt from the beginning of the generated text
            if generated_text.startswith(prompt_text):
                return generated_text[len(prompt_text):].strip()
            return generated_text.strip() # Fallback if prompt not found (e.g. different model behavior)

        except Exception as e:
            logging.error(f"Hugging Face model '{self.model_name}' API call failed: {e}")
            return f"Error communicating with Hugging Face model: {str(e)}"

    def _determine_intent_and_params_with_llm(self, query):
        """
        Uses local Hugging Face model to determine user intent, required MCP, and parameters.
        This is challenging for general models; prompt engineering is key.
        """
        # Prompt needs to be very carefully crafted to guide a general text-gen model.
        # This prompt is an attempt and may need significant iteration for a model like GPT-2.
        prompt = f"""
Instruction: You are an AI assistant. Analyze the user's query and available tools to determine the user's intent and how to fulfill it.
You have access to custom tools (MCPs) and pre-defined configurations.
Respond ONLY with a JSON object matching the specified structure. Do not add any explanations or markdown.

Available Custom MCP Tools:
{json.dumps(CUSTOM_MCP_MODULES, indent=2)}

Available Pre-defined MCP Configurations (examples):
{json.dumps(self.mcp_configurations[:1], indent=2)} 

Factual Memory Keys: {list(self.factual_memory.keys())}
Procedural Memory Keys: {list(self.procedural_memory.keys())}

JSON Output Structure:
{{
  "intent": "Description of user's goal.",
  "mcp_tool_name": "Name of custom MCP tool or 'None'.",
  "mcp_config_name": "Name of pre-defined MCP configuration or 'None'.",
  "parameters": {{ "param1": "value1", ... }},
  "requires_further_clarification": true/false,
  "clarification_question": "Question if clarification is needed, else 'None'.",
  "response_to_user": "A direct, friendly response to the user."
}}

Example:
User Query: "I need to process sensor data: '[{{\"id\":\"s1\", \"val\":10}}]'"
JSON Output:
{{
  "intent": "Process real-time sensor data",
  "mcp_tool_name": "RealTimeStreamingMCP",
  "mcp_config_name": null,
  "parameters": {{ "raw_data_json": "[{{\\"id\\":\\"s1\\", \\"val\\":10}}]" }},
  "requires_further_clarification": false,
  "clarification_question": "None",
  "response_to_user": "I will process your sensor data using the RealTimeStreamingMCP."
}}

User Query: "{query}"
JSON Output:
"""
        # Max length needs to be enough for the JSON.
        # Temperature close to 0 for more deterministic output.
        llm_response_str = self._get_huggingface_completion(prompt, max_length=300, temperature=0.2, top_p=0.7)
        
        try:
            # Try to extract JSON block if it's embedded or has trailing text.
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start:
                llm_response_str = llm_response_str[json_start:json_end]
            else: # If no clear JSON block, assume it's a malformed attempt or natural language
                raise json.JSONDecodeError("No valid JSON block found in LLM response.", llm_response_str, 0)


            parsed_response = json.loads(llm_response_str)
            logging.info(f"LLM Intent Response (parsed): {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: '{llm_response_str}'. Error: {e}")
            # Fallback for when the local model fails to generate valid JSON
            return {
                "intent": "Error: LLM failed to generate structured response.", 
                "mcp_tool_name": None, 
                "mcp_config_name": None,
                "parameters": {}, 
                "requires_further_clarification": True,
                "clarification_question": "I had trouble understanding your request with the local model. Could you be more specific or simplify your query?",
                "response_to_user": f"I'm having difficulty with that request using my current model. The raw response was: '{llm_response_str[:100]}...'. Could you try rephrasing?"
            }

    def use_custom_mcp(self, mcp_name, method_name, params):
        # Identical to previous agents
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
                else: resolved_params[key] = value
        elif params is not None: resolved_params = params
        try:
            method_to_call = getattr(mcp_instance, method_name)
            logging.info(f"Calling MCP '{mcp_name}', method '{method_name}' with params: {resolved_params}")
            if isinstance(resolved_params, dict): result = method_to_call(**resolved_params)
            elif resolved_params is not None: result = method_to_call(resolved_params)
            else: result = method_to_call()
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
        # Largely identical to previous agents
        logging.info(f"\n--- Processing Query: '{query}' ---")
        intent_data = self._determine_intent_and_params_with_llm(query)
        # TODO: A2A Delegation Point: Consider if task should be delegated via self.a2a_client.
        response_to_user = intent_data.get("response_to_user", "I'm processing your request using the Hugging Face model.")
        
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
                response_to_user = f"I identified {mcp_tool_name} but unsure which action. Specificity needed."
                self.add_procedural_memory(f"Query for {mcp_tool_name}", [query, f"Uncertain on method for {mcp_tool_name}"])
            else:
                mcp_result = self.use_custom_mcp(mcp_name=mcp_tool_name, method_name=method_to_call, params=params.get("mcp_params", params))
                response_to_user = f"Used {mcp_tool_name}: {str(mcp_result)[:200]}..."
                self.add_factual_memory(f"last_mcp_result_{mcp_tool_name}", str(mcp_result)[:500])
        elif mcp_config_name and mcp_config_name != "None":
            selected_config = next((c for c in self.mcp_configurations if c.get("name") == mcp_config_name), None)
            if selected_config:
                response_to_user = f"Found MCP config: '{mcp_config_name}'. Details: {selected_config['protocol_details']}. How to use?"
                self.add_factual_memory("last_selected_mcp_config", selected_config)
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Identified config: {mcp_config_name}"])
            else:
                response_to_user = f"Could not find details for MCP config: '{mcp_config_name}'."
                self.add_procedural_memory(f"Query for MCP Config", [query, f"Failed to find config: {mcp_config_name}"])
        elif not (mcp_tool_name and mcp_tool_name != "None") and not (mcp_config_name and mcp_config_name != "None") and intent_data.get("response_to_user"):
             response_to_user = intent_data.get("response_to_user")
             self.add_procedural_memory("General query", [query, f"LLM response: {response_to_user}"])
        
        if not mcp_tool_name and not mcp_config_name and not intent_data.get("requires_further_clarification") and response_to_user == "I'm processing your request using the Hugging Face model.":
            logging.info("No tool/config used, LLM provided no direct response. General completion fallback.")
            fallback_prompt = f"Question: {query}\nAnswer:"
            response_to_user = self._get_huggingface_completion(fallback_prompt, max_length=100, temperature=0.5)
            self.add_procedural_memory("General query fallback", [query, f"HF fallback response: {response_to_user}"])

        logging.info(f"Final Response: {response_to_user}")
        return response_to_user

    def get_agent_card(self) -> AgentCard:
        """Constructs and returns the AgentCard for this agent."""
        capabilities = []
        # Try to use CUSTOM_MCP_MODULES if defined and populated
        if hasattr(self, 'custom_mcps') and self.custom_mcps:
            for mcp_name, mcp_instance in self.custom_mcps.items():
                docstring = mcp_instance.__doc__.splitlines()[0].strip() if mcp_instance.__doc__ else f"Default capability for {mcp_name}"
                params_schema = []
                default_methods = {
                    "RealTimeStreamingMCP": "process_stream_data",
                    "FederatedLearningMCP": "get_global_model",
                    "RobustMCP": "process_message"
                }
                method_name = default_methods.get(mcp_name, "execute")

                if mcp_name == "RealTimeStreamingMCP" and hasattr(mcp_instance, "process_stream_data"):
                    params_schema.append({"name": "raw_data_json", "type": "string", "required": True, "description": "JSON string of the raw data stream."})
                    capability_name = f"{mcp_name}_process_stream_data"
                    description = f"Processes real-time streaming data using {mcp_name}."
                elif mcp_name == "FederatedLearningMCP" and hasattr(mcp_instance, "get_global_model"):
                    capability_name = f"{mcp_name}_get_global_model"
                    description = f"Retrieves the global model using {mcp_name}."
                elif mcp_name == "RobustMCP" and hasattr(mcp_instance, "process_message"):
                    params_schema.append({"name": "message", "type": "object", "required": True, "description": "The message object to process."})
                    capability_name = f"{mcp_name}_process_message"
                    description = f"Processes a message robustly using {mcp_name}."
                else:
                    capability_name = f"{mcp_name}_{method_name}"
                    description = docstring
                    params_schema.append({"name": "params", "type": "object", "required": False, "description": f"Parameters for {method_name} of {mcp_name}."})

                capabilities.append({
                    "name": capability_name, "description": description,
                    "parameters": params_schema,
                    "returns": {"type": "object", "description": f"Result of {capability_name} execution."}
                })
        else:
            # Fallback for HuggingFaceAgent if no MCPs or if they are not primary
            # Define a generic capability based on its core task
            primary_capability_name = f"{self.task.replace('-', '_')}_with_{self.model_name.replace('/', '_')}"
            description = f"Performs {self.task} using the {self.model_name} model."
            params_schema = [{"name": "prompt", "type": "string", "required": True, "description": "Input text/prompt for the model."}]
            if self.task == "text2text-generation" or self.task == "summarization": # Tasks that take text and return text
                 returns_schema = {"type": "string", "description": "Generated text."}
            elif self.task == "text-generation":
                 returns_schema = {"type": "string", "description": "Generated text continuation."}
            else: # Default if task is different
                 returns_schema = {"type": "object", "description": "Output from the model."}

            capabilities.append({
                "name": primary_capability_name,
                "description": description,
                "parameters": params_schema,
                "returns": returns_schema
            })

        return AgentCard(
            agent_id=self.agent_id,
            name=f"HuggingFace Transformer Agent ({self.model_name})",
            description=f"An agent providing access to HuggingFace Transformer models, currently configured for {self.task} with {self.model_name}.",
            version="1.0.0",
            capabilities=capabilities,
            endpoint_url=self.a2a_endpoint_url,
            documentation_url=f"https://huggingface.co/{self.model_name}" if '/' in self.model_name else f"https://huggingface.co/models?search={self.model_name}"
        )

    def handle_a2a_task_request(self, task_data: dict) -> dict:
        """
        Handles an incoming A2A task request.
        Simulates task execution for now.
        """
        logging.info(f"Received A2A task request in HuggingFaceAgent: {json.dumps(task_data, indent=2)}")
        try:
            task = A2ATask.from_dict(task_data)
            logging.info(f"Parsed A2A Task in HuggingFaceAgent: {task.to_json()}")

            agent_capabilities = self.get_agent_card().capabilities
            is_capable = any(cap['name'] == task.capability_name for cap in agent_capabilities)

            if is_capable:
                logging.info(f"Simulating execution of A2A capability: {task.capability_name} with params: {task.parameters} in {self.__class__.__name__}")
                # TODO: Implement actual task execution logic.
                # This might involve calling self.hf_pipeline directly if it's a generic model task,
                # or self.use_custom_mcp if it's an MCP-based capability.
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "taskId": task.task_id,
                        "status": "completed",
                        "artifacts": [{"type": "text", "content": f"Successfully executed {task.capability_name} via {self.__class__.__name__}"}]
                    },
                    "id": task_data.get("id")
                }
            else:
                logging.warning(f"Capability '{task.capability_name}' not found for A2A task {task.task_id} in {self.__class__.__name__}.")
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found", "data": f"Capability '{task.capability_name}' not implemented by {self.__class__.__name__} {self.agent_id}."},
                    "id": task_data.get("id")
                }
        except Exception as e:
            logging.error(f"Error handling A2A task in {self.__class__.__name__}: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": task_data.get("id")
            }

# Example Usage
if __name__ == '__main__':
    # For CPU, device=-1. For GPU, device=0 (if CUDA is available).
    # Using a smaller model like 'gpt2' or 'distilgpt2' for broader compatibility without specific hardware.
    # 'google/flan-t5-small' or 'facebook/bart-large-cnn' could also be options if 'gpt2' struggles too much with JSON.
    # For this demo, we'll stick with 'gpt2' and acknowledge its limitations for structured JSON.
    HF_MODEL = os.environ.get("HF_MODEL_NAME", "gpt2") 
    HF_DEVICE = int(os.environ.get("HF_DEVICE", -1)) # Default to CPU

    print(f"--- HuggingFaceAgent Demo (Model: {HF_MODEL}, Device: {'GPU' if HF_DEVICE==0 else 'CPU'}) ---")
    
    agent_specific_mcp_configs = {
        "streaming_mcp_config": {'validation_rules': {'value': {'type': float}}},
        "federated_mcp_config": {"global_model_id": "hf_agent_demo_model_v1"},
        "robust_mcp_config": {'max_retries': 1, 'base_retry_delay_seconds': 0.1}
    }

    try:
        agent = HuggingFaceAgent(
            model_name=HF_MODEL, # e.g., "gpt2", "distilgpt2", or a more capable small model
            task="text-generation",
            agent_config=agent_specific_mcp_configs,
            device=HF_DEVICE 
        )

        agent.add_factual_memory("user_goal", "evaluate local LLM agent")
        print(f"User's goal: {agent.get_factual_memory('user_goal')}")

        agent.add_procedural_memory("hf_setup", ["Load model", "Initialize pipeline", "Test inference"])
        print(f"HF Setup: {agent.get_procedural_memory('hf_setup')}")

        queries = [
            "Hello, what are your capabilities?",
            "I have sensor data to stream: '[{\"id\":\"temp_sensor\", \"value\":25.5}]'",
            "What was the last result from RobustMCP?", # Test memory (LLM might struggle to link this without explicit instruction)
            "How do I submit a message '{\"id\":\"test01\", \"payload\":{\"info\":\"some data\"}}' using the robust MCP?",
            "Tell me about the 'Standard HTTPS/JSON MCP' configuration."
        ]

        for q_idx, q in enumerate(queries):
            print(f"\n--- Query {q_idx+1} ---")
            response = agent.process_query(q)
            print(f"User Query: {q}\nAgent Response: {response}")
            # No API calls to external services, so shorter sleep is fine.
            # Model inference itself can be slow on CPU.
            time.sleep(1) 

        print("\n--- Final Memory States ---")
        print(f"Factual Memory: {json.dumps(agent.factual_memory, indent=2)}")
        print(f"Procedural Memory: {json.dumps(agent.procedural_memory, indent=2)}")

            print("\n--- HuggingFace Agent Card ---")
            agent_card = agent.get_agent_card()
            print(agent_card.to_json())

            print("\n--- HuggingFace A2A Task Handling Simulation ---")
            # Determine a valid capability name from the agent card for the test
            # Fallback to a generic one if MCPs are not the primary focus or empty
            hf_capabilities = agent_card.capabilities
            sim_capability_name = "generate_text_with_hf_model" # A generic default
            if hf_capabilities:
                 # Try to pick one of the MCP based capabilities if they exist
                mcp_cap = next((c['name'] for c in hf_capabilities if "MCP" in c['name']), None)
                if mcp_cap:
                    sim_capability_name = mcp_cap
                else: # If no MCP caps, pick the first one (likely the generic model task)
                    sim_capability_name = hf_capabilities[0]['name']


            sample_a2a_task_data = {
                "taskId": "a2a-hf-task-001",
                "clientAgentId": "test-client-for-hf",
                "remoteAgentId": agent.agent_id,
                "capabilityName": sim_capability_name,
                "parameters": {"prompt": "Test A2A to HuggingFace"} if "generate_text" in sim_capability_name else {"message": {"id":"hf01", "payload":"data"}} if "RobustMCP" in sim_capability_name else {"raw_data_json": "[]"} if "StreamingMCP" in sim_capability_name else {}
            }
            a2a_response = agent.handle_a2a_task_request(sample_a2a_task_data)
            print(f"A2A Task Response (HuggingFace): {json.dumps(a2a_response, indent=2)}")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except ImportError as ie:
        print(f"Import Error: {ie}. Make sure 'transformers' and 'torch' (or 'tensorflow') are installed.")
    except Exception as e:
        print(f"An unexpected error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- End of HuggingFaceAgent Demo ---")

import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys
import tempfile

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import the agent
from huggingface_agent import HuggingFaceAgent # Changed import

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestHuggingFaceAgent(unittest.TestCase): # Changed class name

    def setUp(self):
        """Set up for each test."""
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        self.patches = {
            # Patch transformers.pipeline
            'hf_pipeline_constructor': patch('huggingface_agent.pipeline'), 
            'factual_memory_path': patch('huggingface_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('huggingface_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            'mcp_config_path': patch('huggingface_agent.MCP_CONFIG_FILE', ACTUAL_MCP_CONFIG_FILE),
        }

        self.mock_hf_pipeline_constructor = self.patches['hf_pipeline_constructor'].start()
        self.mock_hf_pipeline_instance = MagicMock() # This is what pipeline() returns
        self.mock_hf_pipeline_constructor.return_value = self.mock_hf_pipeline_instance
        
        # Mock tokenizer pad_token_id for the pipeline instance
        self.mock_hf_pipeline_instance.tokenizer = MagicMock()
        self.mock_hf_pipeline_instance.tokenizer.pad_token_id = 50256 # Example EOS token for GPT-2
        self.mock_hf_pipeline_instance.model = MagicMock()
        self.mock_hf_pipeline_instance.model.config = MagicMock()
        self.mock_hf_pipeline_instance.model.config.eos_token_id = 50256


        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()
        self.patches['mcp_config_path'].start()

        self.agent_mcp_configs = {
            "streaming_mcp_config": {'validation_rules': {'id': {'type': str}}},
            "federated_mcp_config": {"global_model_id": "hf_test_model"}, # Differentiation
            "robust_mcp_config": {'max_retries': 1, 'base_retry_delay_seconds': 0.01}
        }

        self.agent = HuggingFaceAgent( # Changed agent instantiation
            model_name="gpt2", # Default, or a specific test model
            task="text-generation",
            agent_config=self.agent_mcp_configs,
            device=-1 # CPU for tests
        )
        self.assertTrue(len(self.agent.custom_mcps) > 0, "Custom MCPs were not loaded")


    def tearDown(self):
        """Clean up after each test."""
        for patcher in self.patches.values():
            patcher.stop()
        os.remove(self.temp_factual_mem_file.name)
        os.remove(self.temp_procedural_mem_file.name)

    def test_initialization_and_empty_memory_load(self):
        self.assertIsNotNone(self.agent.hf_pipeline)
        self.mock_hf_pipeline_constructor.assert_called_once_with(
            "text-generation", model="gpt2", tokenizer="gpt2", device=-1
        )
        self.assertEqual(self.agent.factual_memory, {})
        self.assertEqual(self.agent.procedural_memory, {})
        self.assertTrue(len(self.agent.mcp_configurations) > 0)

    def test_memory_load_existing(self):
        factual_data = {"fact_hf": "value_hf"} # Differentiation
        procedural_data = {"proc_hf": {"steps": ["step_hf"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        agent_reloaded = HuggingFaceAgent(
            model_name="gpt2", agent_config=self.agent_mcp_configs, device=-1
        )
        self.assertEqual(agent_reloaded.factual_memory, factual_data)
        self.assertEqual(agent_reloaded.procedural_memory["proc_hf"]["steps"], procedural_data["proc_hf"]["steps"])

    def test_factual_memory_operations(self):
        self.agent.add_factual_memory("test_fact_hf_key", "test_fact_hf_value")
        self.assertEqual(self.agent.get_factual_memory("test_fact_hf_key"), "test_fact_hf_value")
        with open(self.temp_factual_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_fact_hf_key"], "test_fact_hf_value")

    def test_procedural_memory_operations(self):
        self.agent.add_procedural_memory("test_proc_hf_task", ["step_gamma", "step_delta"])
        retrieved_proc = self.agent.get_procedural_memory("test_proc_hf_task")
        self.assertEqual(retrieved_proc["steps"], ["step_gamma", "step_delta"])
        with open(self.temp_procedural_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_proc_hf_task"]["steps"], ["step_gamma", "step_delta"])

    def _mock_hf_pipeline_response(self, prompt_text, generated_text_suffix):
        """Helper to mock the Hugging Face pipeline response."""
        # The pipeline returns a list of dicts, each with 'generated_text'
        # The agent's _get_huggingface_completion tries to strip the prompt.
        full_generated_text = prompt_text + generated_text_suffix
        self.mock_hf_pipeline_instance.return_value = [{'generated_text': full_generated_text}]

    def test_llm_intent_parsing_ideal_json(self):
        """Test intent parsing when HF model ideally returns perfect JSON."""
        mock_llm_json_output_str = json.dumps({
            "intent": "HF Test Intent Ideal",
            "mcp_tool_name": "RobustMCP", 
            "mcp_config_name": None,
            "parameters": {"message_json": "{\"id\":\"hf_msg1\"}", "method_name": "process_message"},
            "requires_further_clarification": False,
            "clarification_question": "None", # Ensure it's 'None' not Python None for JSON consistency
            "response_to_user": "Processing your HF message with RobustMCP."
        })
        
        # The prompt passed to _get_huggingface_completion is complex. We need to mock based on that.
        # For this test, we'll assume the _get_huggingface_completion is called and mock its direct output.
        # This simplifies the test setup.
        with patch.object(self.agent, '_get_huggingface_completion', return_value=mock_llm_json_output_str) as mock_get_completion:
            parsed_intent_data = self.agent._determine_intent_and_params_with_llm("some query for hf agent")
            mock_get_completion.assert_called_once() # Verify it was called

        self.assertEqual(parsed_intent_data["intent"], "HF Test Intent Ideal")
        self.assertEqual(parsed_intent_data["mcp_tool_name"], "RobustMCP")
        self.assertEqual(parsed_intent_data["parameters"]["message_json"], "{\"id\":\"hf_msg1\"}")


    def test_llm_intent_parsing_malformed_json(self):
        """Test intent parsing when HF model returns malformed/incomplete JSON."""
        malformed_json_str = "{ \"intent\": \"HF Malformed\", \"mcp_tool_name\": \"RealTimeStreamingMCP\"," # Missing closing parts
        
        with patch.object(self.agent, '_get_huggingface_completion', return_value=malformed_json_str) as mock_get_completion:
            parsed_intent_data = self.agent._determine_intent_and_params_with_llm("another query")
            mock_get_completion.assert_called_once()

        self.assertEqual(parsed_intent_data["intent"], "Error: LLM failed to generate structured response.")
        self.assertTrue(parsed_intent_data["requires_further_clarification"])
        self.assertIn("I had trouble understanding your request with the local model.", parsed_intent_data["clarification_question"])


    @patch('custom_mcp_3.RobustMCP.process_message') # Patching RobustMCP method
    def test_tool_usage_robust_mcp_with_hf(self, mock_robust_process_method):
        # Simulate LLM providing perfect JSON for this tool use case
        mock_llm_json_output = {
            "intent": "Process robust message with HF",
            "mcp_tool_name": "RobustMCP",
            "mcp_config_name": None,
            "parameters": {"message_json": "{\"id\":\"hf_robust\"}", "method_name": "process_message"},
            "requires_further_clarification": False,
            "response_to_user": "HF using RobustMCP for your message."
        }
        with patch.object(self.agent, '_determine_intent_and_params_with_llm', return_value=mock_llm_json_output):
            mock_robust_process_method.return_value = "HF robust message processed"
            response = self.agent.process_query("process this hf robust message '{\"id\":\"hf_robust\"}'")
        
        mock_robust_process_method.assert_called_once_with(message_json="{\"id\":\"hf_robust\"}")
        self.assertIn("Used RobustMCP: HF robust message processed", response)


    def test_llm_pipeline_error_handling(self):
        """Test agent's behavior when HuggingFace pipeline call fails."""
        self.mock_hf_pipeline_instance.side_effect = Exception("HF Pipeline Error")
        
        # This error occurs in _get_huggingface_completion, which is called by _determine_intent_and_params_with_llm
        parsed_intent_data = self.agent._determine_intent_and_params_with_llm("query causing pipeline error")
        
        self.assertEqual(parsed_intent_data["intent"], "Error: LLM failed to generate structured response.")
        self.assertTrue(parsed_intent_data["requires_further_clarification"])
        # The response_to_user from this fallback will contain the raw error.
        self.assertIn("Error communicating with Hugging Face model: HF Pipeline Error", parsed_intent_data["response_to_user"])

        # Test the full process_query flow
        response = self.agent.process_query("query causing pipeline error")
        self.assertIn("Error communicating with Hugging Face model: HF Pipeline Error", response)


    def test_mcp_config_selection_flow(self):
        # Assuming "Standard HTTPS/JSON MCP" exists
        config_name_to_test = "Standard HTTPS/JSON MCP" 
        mock_llm_json_output = {
            "intent": "Inquire about standard MCP config with HF",
            "mcp_tool_name": None,
            "mcp_config_name": config_name_to_test,
            "parameters": {},
            "requires_further_clarification": False,
            "response_to_user": f"HF found info about {config_name_to_test}."
        }
        with patch.object(self.agent, '_determine_intent_and_params_with_llm', return_value=mock_llm_json_output):
            response = self.agent.process_query(f"Tell me about the {config_name_to_test} using HF")
        
        self.assertIn(f"Found MCP config: '{config_name_to_test}'", response)
        self.assertIn("HTTPS", response) 
        self.assertIsNotNone(self.agent.get_factual_memory("last_selected_mcp_config"))
        self.assertEqual(self.agent.get_factual_memory("last_selected_mcp_config")["name"], config_name_to_test)

    def test_clarification_flow(self):
        clarification_q = "What is the participant ID for the federated learning model update?"
        mock_llm_json_output = {
            "intent": "FL update via HF",
            "mcp_tool_name": "FederatedLearningMCP",
            "parameters": {"method_name": "submit_model_update"}, # Missing params
            "requires_further_clarification": True,
            "clarification_question": clarification_q,
            "response_to_user": clarification_q
        }
        with patch.object(self.agent, '_determine_intent_and_params_with_llm', return_value=mock_llm_json_output):
            query = "I want to submit an FL update using HF."
            response = self.agent.process_query(query)

        self.assertEqual(response, clarification_q)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_for_query"), query)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_question"), clarification_q)

if __name__ == '__main__':
    unittest.main()

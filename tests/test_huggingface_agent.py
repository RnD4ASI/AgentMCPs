import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import json
import os
import sys
import tempfile
import importlib # For mocking importlib.util

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_agent import HuggingFaceAgent # Changed agent

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestHuggingFaceAgentMCP(unittest.TestCase): # Renamed for clarity

    def setUp(self):
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        if not os.path.exists(ACTUAL_MCP_CONFIG_FILE):
            minimal_mcp_config = {"mcp_version": "1.0.0", "models": [], "skills": [], "context_types": []}
            with open(ACTUAL_MCP_CONFIG_FILE, 'w') as f: json.dump(minimal_mcp_config, f)
            self.created_minimal_mcp_config = True
        else:
            self.created_minimal_mcp_config = False

        self.patches = {
            'hf_pipeline_constructor': patch('huggingface_agent.pipeline'),
            'factual_memory_path': patch('huggingface_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('huggingface_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
        }

        self.mock_hf_pipeline_constructor = self.patches['hf_pipeline_constructor'].start()
        self.mock_hf_pipeline_instance = MagicMock()
        self.mock_hf_pipeline_constructor.return_value = self.mock_hf_pipeline_instance

        # Mock tokenizer and model config for pad_token_id logic in agent's __init__
        self.mock_hf_pipeline_instance.tokenizer = MagicMock()
        self.mock_hf_pipeline_instance.tokenizer.pad_token_id = 50256 # Example
        self.mock_hf_pipeline_instance.model = MagicMock()
        self.mock_hf_pipeline_instance.model.config = MagicMock()
        self.mock_hf_pipeline_instance.model.config.eos_token_id = 50256

        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()

        self.agent = HuggingFaceAgent( # Changed agent instantiation
            model_name="gpt2", # Default test model
            task="text-generation",
            device=-1 # CPU for tests
        )

    def tearDown(self):
        for patcher in self.patches.values():
            patcher.stop()
        os.remove(self.temp_factual_mem_file.name)
        os.remove(self.temp_procedural_mem_file.name)
        if self.created_minimal_mcp_config and os.path.exists(ACTUAL_MCP_CONFIG_FILE):
            # os.remove(ACTUAL_MCP_CONFIG_FILE)
            pass

    def test_mcp_registry_loading(self):
        self.assertIsNotNone(self.agent.mcp_registry)
        self.assertIn("skills_map", self.agent.mcp_registry)
        self.assertIn("realtime_data_processor", self.agent.mcp_registry["skills_map"])

    def _mock_hf_pipeline_call_response(self, prompt_text_sent_to_pipeline, generated_suffix_str):
        # The agent's _get_huggingface_completion strips the prompt_text_sent_to_pipeline
        # from the pipeline's output.
        full_generated_text = prompt_text_sent_to_pipeline + generated_suffix_str
        self.mock_hf_pipeline_instance.return_value = [{'generated_text': full_generated_text}]

    def test_llm_determines_skill_id(self):
        mock_llm_json_output_suffix = json.dumps({ # This is the part *after* the prompt
            "intent": "HF skill test for robust messaging",
            "skill_id": "robust_message_handler",
            "parameters": {"message_json": "{\"hf_id\":\"msg_hf\"}"},
            "requires_further_clarification": False,
            "clarification_question": "None",
            "response_to_user": "HF handling message robustly."
        })

        # We need to know the exact prompt string passed to the pipeline to mock effectively.
        # This is complex as the prompt is constructed in _determine_intent_and_params_with_llm.
        # For a more focused test on parsing, we can directly mock _get_huggingface_completion.
        with patch.object(self.agent, '_get_huggingface_completion', return_value=mock_llm_json_output_suffix) as mock_get_hf_completion:
            intent_data = self.agent._determine_intent_and_params_with_llm("Handle a message via HF")
            # Check that _get_huggingface_completion was called (it takes the constructed prompt)
            mock_get_hf_completion.assert_called_once()
            # The first argument to the call is the actual prompt string.
            # We can inspect mock_get_hf_completion.call_args[0][0] if needed.

        self.assertEqual(intent_data["skill_id"], "robust_message_handler")
        self.assertEqual(intent_data["parameters"]["message_json"], "{\"hf_id\":\"msg_hf\"}")

    def test_prepare_skill_context_all_memory_snapshots(self):
        self.agent.add_factual_memory("hf_fact", "hf_value")
        self.agent.add_procedural_memory("hf_proc", ["hf_step1"])

        mock_skill_def = {
            "id": "skill_all_mem",
            "expected_context": ["factual_memory_snapshot", "procedural_memory_snapshot"]
        }
        with patch.dict(self.agent.mcp_registry['skills_map'], {'skill_all_mem': mock_skill_def}):
            context_data = self.agent._prepare_skill_context("skill_all_mem")

        self.assertIn("factual_memory_snapshot", context_data)
        self.assertEqual(context_data["factual_memory_snapshot"]["hf_fact"], "hf_value")
        self.assertIn("procedural_memory_snapshot", context_data)
        self.assertEqual(context_data["procedural_memory_snapshot"]["hf_proc"]["steps"], ["hf_step1"])
        self.assertIn("agent_name", context_data)
        self.assertEqual(context_data["agent_name"], "HuggingFaceAgent")


    @patch('huggingface_agent.HuggingFaceAgent.invoke_mcp_skill')
    def test_process_query_uses_skill(self, mock_invoke_skill):
        mock_llm_json_output = {
            "intent": "Invoke skill with HF",
            "skill_id": "realtime_data_processor", # A known skill from your config
            "parameters": {"raw_data_json": "[{\"id\":\"hf_data\"}]"},
            "requires_further_clarification": False,
            "response_to_user": "HF will process data."
        }
        # Mocking _determine_intent_and_params_with_llm directly for simplicity here
        with patch.object(self.agent, '_determine_intent_and_params_with_llm', return_value=mock_llm_json_output):
            mock_skill_response = {"status": "success", "data": {"processed_ok": True}}
            mock_invoke_skill.return_value = mock_skill_response

            with patch.object(self.agent, '_prepare_skill_context', return_value={"hf_ctx": "prepared"}) as mock_prepare_context:
                response = self.agent.process_query("Process HF data")

            mock_prepare_context.assert_called_once_with("realtime_data_processor")
            mock_invoke_skill.assert_called_once_with(
                "realtime_data_processor",
                {"raw_data_json": "[{\"id\":\"hf_data\"}]"},
                {"hf_ctx": "prepared"}
            )
            self.assertIn("Skill 'realtime_data_processor' executed successfully.", response)
            self.assertIn(json.dumps({"processed_ok": True}), response)

    @patch('importlib.util')
    @patch.dict(sys.modules, {})
    def test_invoke_mcp_skill_successful_invocation(self, mock_importlib_util):
        skill_id = "robust_message_handler"

        mock_skill_module = MagicMock()
        mock_skill_class_constructor = MagicMock()
        mock_skill_instance = MagicMock()
        mock_skill_method = MagicMock(return_value={"status": "success", "data": "hf robust skill works"})

        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        mock_importlib_util.module_from_spec.return_value = mock_skill_module

        # Handler path from mcp_configurations.json for "robust_message_handler" is "RobustMCP.process_message"
        setattr(mock_skill_module, "RobustMCP", mock_skill_class_constructor)
        mock_skill_class_constructor.return_value = mock_skill_instance
        setattr(mock_skill_instance, "process_message", mock_skill_method)

        parameters = {"message_json": "hf_message_data"}
        context = {"caller": "hf_agent_test"}

        self.assertTrue(skill_id in self.agent.mcp_registry['skills_map'])
        skill_response = self.agent.invoke_mcp_skill(skill_id, parameters, context)

        mock_importlib_util.spec_from_file_location.assert_called_once_with(
            self.agent.mcp_registry['skills_map'][skill_id]['handler_module'],
            f"{self.agent.mcp_registry['skills_map'][skill_id]['handler_module']}.py"
        )
        mock_skill_method.assert_called_once_with(parameters, context)
        self.assertEqual(skill_response["status"], "success")
        self.assertEqual(skill_response["data"], "hf robust skill works")


    def test_llm_parsing_error_flow(self):
        """Test how agent handles LLM response that is not valid JSON."""
        # Simulate _get_huggingface_completion returning non-JSON string
        with patch.object(self.agent, '_get_huggingface_completion', return_value="This is not JSON.") as mock_get_hf_completion:
            response = self.agent.process_query("A query leading to bad JSON")

        self.assertIn("My local model provided a response I couldn't parse.", response)
        self.assertIn("This is not JSON.", response) # Agent includes snippet of bad response.


    def test_initialization_loads_existing_memory(self):
        factual_data = {"fact_hf_mcp": "val_hf_mcp"}
        procedural_data = {"proc_hf_mcp": {"steps": ["step_mcp_hf"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        reloaded_agent = HuggingFaceAgent(model_name="gpt2", device=-1)
        self.assertEqual(reloaded_agent.factual_memory, factual_data)
        self.assertEqual(reloaded_agent.procedural_memory["proc_hf_mcp"]["steps"], procedural_data["proc_hf_mcp"]["steps"])

if __name__ == '__main__':
    unittest.main()

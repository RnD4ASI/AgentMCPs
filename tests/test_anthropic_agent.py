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

from anthropic_agent import AnthropicAgent # Changed agent

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestAnthropicAgentMCP(unittest.TestCase): # Renamed for clarity

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
            'Anthropic_client': patch('anthropic_agent.Anthropic'), # Patched Anthropic client
            'factual_memory_path': patch('anthropic_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('anthropic_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
        }

        self.mock_anthropic_client_constructor = self.patches['Anthropic_client'].start()
        self.mock_anthropic_client_instance = MagicMock()
        self.mock_anthropic_client_constructor.return_value = self.mock_anthropic_client_instance

        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()

        self.agent = AnthropicAgent( # Changed agent instantiation
            api_key="dummy_anthropic_key",
            model_name="claude-3-haiku-20240307"
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
        # Check if a known skill ID (as defined in your mcp_configurations.json) is in the map
        self.assertIn("robust_message_handler", self.agent.mcp_registry["skills_map"])

    def _mock_llm_completion_response(self, response_text_str):
        # Anthropic's response structure: response.content is a list of blocks.
        mock_text_block = MagicMock()
        mock_text_block.text = response_text_str
        mock_completion_response = MagicMock()
        mock_completion_response.content = [mock_text_block]
        self.mock_anthropic_client_instance.messages.create.return_value = mock_completion_response

    def test_llm_determines_skill_id(self):
        mock_llm_output = {
            "intent": "Robust message handling",
            "skill_id": "robust_message_handler",
            "parameters": {"message_json": "{\"id\":\"anthropic_msg\"}"},
            "requires_further_clarification": False,
            "clarification_question": "None",
            "response_to_user": "Handling your message robustly via Anthropic."
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_output))

        intent_data = self.agent._determine_intent_and_params_with_llm("Handle a message for me")
        self.assertEqual(intent_data["skill_id"], "robust_message_handler")
        self.assertEqual(intent_data["parameters"]["message_json"], "{\"id\":\"anthropic_msg\"}")

    def test_prepare_skill_context_procedural_snapshot(self):
        self.agent.add_procedural_memory("anthropic_task", ["step_claude"])
        # Skill definition that expects procedural memory
        mock_skill_def = {"id": "skill_with_proc_mem", "expected_context": ["procedural_memory_snapshot"]}
        with patch.dict(self.agent.mcp_registry['skills_map'], {'skill_with_proc_mem': mock_skill_def}):
            context_data = self.agent._prepare_skill_context("skill_with_proc_mem")

        self.assertIn("procedural_memory_snapshot", context_data)
        self.assertEqual(context_data["procedural_memory_snapshot"]["anthropic_task"]["steps"], ["step_claude"])
        self.assertIn("agent_name", context_data)
        self.assertEqual(context_data["agent_name"], "AnthropicAgent")


    @patch('anthropic_agent.AnthropicAgent.invoke_mcp_skill')
    def test_process_query_uses_skill(self, mock_invoke_skill):
        mock_llm_output = {
            "intent": "Stream data via Anthropic",
            "skill_id": "realtime_data_processor",
            "parameters": {"raw_data_json": "[{}]"},
            "requires_further_clarification": False,
            "response_to_user": "Anthropic will stream data."
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_output))

        mock_skill_response = {"status": "success", "data": {"streamed_count": 1}}
        mock_invoke_skill.return_value = mock_skill_response

        with patch.object(self.agent, '_prepare_skill_context', return_value={"ctx_prepared": "yes"}) as mock_prepare_context:
            response = self.agent.process_query("Stream this data")

        mock_prepare_context.assert_called_once_with("realtime_data_processor")
        mock_invoke_skill.assert_called_once_with(
            "realtime_data_processor",
            {"raw_data_json": "[{}]"},
            {"ctx_prepared": "yes"}
        )
        self.assertIn("Skill 'realtime_data_processor' executed successfully.", response)
        self.assertIn(json.dumps({"streamed_count": 1}), response)


    @patch('importlib.util')
    @patch.dict(sys.modules, {})
    def test_invoke_mcp_skill_successful_invocation(self, mock_importlib_util):
        skill_id = "realtime_data_processor"

        mock_skill_module = MagicMock()
        mock_skill_class_constructor = MagicMock()
        mock_skill_instance = MagicMock()
        mock_skill_method = MagicMock(return_value={"status": "success", "data": "anthropic skill executed"})

        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        mock_importlib_util.module_from_spec.return_value = mock_skill_module

        # Handler path from mcp_configurations.json for "realtime_data_processor" is "RealTimeStreamingMCP.process_stream_data"
        # The skill file realtime_processing_skill.py defines class RealTimeStreamingMCP
        setattr(mock_skill_module, "RealTimeStreamingMCP", mock_skill_class_constructor)
        mock_skill_class_constructor.return_value = mock_skill_instance
        setattr(mock_skill_instance, "process_stream_data", mock_skill_method)

        parameters = {"raw_data_json": "anthropic_data"}
        context = {"user_model_preference": "claude"}

        self.assertTrue(skill_id in self.agent.mcp_registry['skills_map'])
        skill_response = self.agent.invoke_mcp_skill(skill_id, parameters, context)

        mock_importlib_util.spec_from_file_location.assert_called_once_with(
            self.agent.mcp_registry['skills_map'][skill_id]['handler_module'],
            f"{self.agent.mcp_registry['skills_map'][skill_id]['handler_module']}.py"
        )
        mock_skill_method.assert_called_once_with(parameters, context)
        self.assertEqual(skill_response["status"], "success")
        self.assertEqual(skill_response["data"], "anthropic skill executed")

    def test_invoke_mcp_skill_definition_missing_handler_details(self):
        skill_id = "skill_missing_handler"
        self.agent.mcp_registry['skills_map'][skill_id] = {
            "id": skill_id, "name": "Test Missing Handler Details",
            "handler_module": None, # Missing
            "handler_class_or_function": None # Missing
        }
        response = self.agent.invoke_mcp_skill(skill_id, {}, {})
        self.assertEqual(response["status"], "error")
        self.assertIn("not configured correctly for invocation", response["error_message"])


    def test_initialization_loads_existing_memory(self):
        factual_data = {"fact_anthropic_mcp": "val_anthropic_mcp"}
        procedural_data = {"proc_anthropic_mcp": {"steps": ["step_mcp_anthropic"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        reloaded_agent = AnthropicAgent(api_key="dummy_key", model_name="claude-3")
        self.assertEqual(reloaded_agent.factual_memory, factual_data)
        self.assertEqual(reloaded_agent.procedural_memory["proc_anthropic_mcp"]["steps"], procedural_data["proc_anthropic_mcp"]["steps"])

if __name__ == '__main__':
    unittest.main()

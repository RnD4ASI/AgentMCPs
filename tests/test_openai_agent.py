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

from openai_agent import OpenAIAgent # Changed agent

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestOpenAIAgentMCP(unittest.TestCase): # Renamed for clarity

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
            'OpenAI_client': patch('openai_agent.OpenAI'), # Patched standard OpenAI client
            'factual_memory_path': patch('openai_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('openai_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
        }

        self.mock_openai_client_constructor = self.patches['OpenAI_client'].start()
        self.mock_openai_client_instance = MagicMock()
        self.mock_openai_client_constructor.return_value = self.mock_openai_client_instance

        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()

        self.agent = OpenAIAgent( # Changed agent instantiation
            api_key="dummy_openai_key",
            model_name="gpt-3.5-turbo"
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
        self.assertIn("federated_model_aggregator", self.agent.mcp_registry["skills_map"]) # Example skill

    def _mock_llm_chat_response(self, response_content_json_str):
        # Same structure as Azure OpenAI for chat completions
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = response_content_json_str
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [mock_choice]
        self.mock_openai_client_instance.chat.completions.create.return_value = mock_completion_response

    def test_llm_determines_skill_id(self):
        mock_llm_output = {
            "intent": "Federated learning action",
            "skill_id": "federated_model_aggregator", # Skill ID from mcp_configurations.json
            "parameters": {"action": "get_global_model"},
            "requires_further_clarification": False,
            "clarification_question": "None",
            "response_to_user": "Fetching global model."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_output))

        intent_data = self.agent._determine_intent_and_params_with_llm("Get federated model")
        self.assertEqual(intent_data["skill_id"], "federated_model_aggregator")
        self.assertEqual(intent_data["parameters"]["action"], "get_global_model")

    def test_prepare_skill_context_memory_snapshots(self):
        self.agent.add_factual_memory("user_preference", "dark_theme")
        # Skill definition that expects factual memory
        mock_skill_def = {"id": "skill_with_fact_mem", "expected_context": ["factual_memory_snapshot"]}
        with patch.dict(self.agent.mcp_registry['skills_map'], {'skill_with_fact_mem': mock_skill_def}):
            context_data = self.agent._prepare_skill_context("skill_with_fact_mem")

        self.assertIn("factual_memory_snapshot", context_data)
        self.assertEqual(context_data["factual_memory_snapshot"]["user_preference"], "dark_theme")
        self.assertIn("agent_name", context_data)
        self.assertEqual(context_data["agent_name"], "OpenAIAgent")


    @patch('openai_agent.OpenAIAgent.invoke_mcp_skill')
    def test_process_query_uses_skill(self, mock_invoke_skill):
        mock_llm_output = {
            "intent": "Handle robust message",
            "skill_id": "robust_message_handler",
            "parameters": {"message_json": "\"{'id':'msg1'}\""},
            "requires_further_clarification": False,
            "response_to_user": "Handling message."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_output))

        mock_skill_response = {"status": "success", "data": {"handled": True}}
        mock_invoke_skill.return_value = mock_skill_response

        with patch.object(self.agent, '_prepare_skill_context', return_value={"context_ok": True}) as mock_prepare_context:
            response = self.agent.process_query("Handle this message robustly")

        mock_prepare_context.assert_called_once_with("robust_message_handler")
        mock_invoke_skill.assert_called_once_with(
            "robust_message_handler",
            {"message_json": "\"{'id':'msg1'}\""},
            {"context_ok": True}
        )
        self.assertIn("Skill 'robust_message_handler' executed successfully.", response)
        self.assertIn(json.dumps({"handled": True}), response)


    @patch('importlib.util')
    @patch.dict(sys.modules, {})
    def test_invoke_mcp_skill_successful_invocation(self, mock_importlib_util):
        skill_id = "robust_message_handler"

        mock_skill_module = MagicMock()
        mock_skill_class_constructor = MagicMock() # Mock for RobustMCP class
        mock_skill_instance = MagicMock()
        mock_skill_method = MagicMock(return_value={"status": "success", "data": "robust skill ok"})

        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        mock_importlib_util.module_from_spec.return_value = mock_skill_module

        # Handler path is "RobustMCP.process_message"
        setattr(mock_skill_module, "RobustMCP", mock_skill_class_constructor)
        mock_skill_class_constructor.return_value = mock_skill_instance
        setattr(mock_skill_instance, "process_message", mock_skill_method)

        parameters = {"message_json": "data"}
        context = {"session_id": "s123"}

        self.assertTrue(skill_id in self.agent.mcp_registry['skills_map'])
        skill_response = self.agent.invoke_mcp_skill(skill_id, parameters, context)

        mock_importlib_util.spec_from_file_location.assert_called_once_with(
            self.agent.mcp_registry['skills_map'][skill_id]['handler_module'],
            f"{self.agent.mcp_registry['skills_map'][skill_id]['handler_module']}.py"
        )
        mock_skill_method.assert_called_once_with(parameters, context)
        self.assertEqual(skill_response["status"], "success")
        self.assertEqual(skill_response["data"], "robust skill ok")

    def test_invoke_mcp_skill_handler_not_found(self):
        skill_id = "skill_with_bad_handler_path"
        self.agent.mcp_registry['skills_map'][skill_id] = {
            "id": skill_id, "name": "Test Bad Handler",
            "handler_module": "realtime_processing_skill", # Valid module
            "handler_class_or_function": "NonExistentClass.non_existent_method" # Invalid path
        }

        # We need to ensure realtime_processing_skill is "importable" by importlib.util.spec_from_file_location
        # This is tricky without actually having the file system reflect this for the test runner.
        # Assuming the module itself can be found by spec_from_file_location due to sys.path.
        # The error should come from getattr trying to find NonExistentClass.

        # To properly test this, we mock spec_from_file_location to return a valid spec,
        # and module_from_spec to return a module object where NonExistentClass is missing.
        with patch('importlib.util.spec_from_file_location') as mock_spec_from_file:
            mock_spec = MagicMock()
            mock_spec.loader = MagicMock()
            mock_spec_from_file.return_value = mock_spec

            with patch('importlib.util.module_from_spec') as mock_module_from_spec:
                mock_skill_module_obj = MagicMock()
                # Make getattr fail when looking for NonExistentClass
                # setattr(mock_skill_module_obj, "NonExistentClass", None) # This makes it exist as None
                # Better to let it raise AttributeError implicitly
                mock_module_from_spec.return_value = mock_skill_module_obj

                response = self.agent.invoke_mcp_skill(skill_id, {}, {})

        self.assertEqual(response["status"], "error")
        self.assertIn("Could not load handler for skill", response["error_message"])
        self.assertIn("NonExistentClass", response["error_message"])


    def test_initialization_loads_existing_memory(self):
        factual_data = {"fact_openai_mcp": "val_openai_mcp"}
        procedural_data = {"proc_openai_mcp": {"steps": ["step_mcp_openai"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        reloaded_agent = OpenAIAgent(api_key="dummy_key", model_name="gpt-3.5-turbo")
        self.assertEqual(reloaded_agent.factual_memory, factual_data)
        self.assertEqual(reloaded_agent.procedural_memory["proc_openai_mcp"]["steps"], procedural_data["proc_openai_mcp"]["steps"])

if __name__ == '__main__':
    unittest.main()

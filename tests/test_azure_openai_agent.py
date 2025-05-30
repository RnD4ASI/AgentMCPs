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

from azure_openai_agent import AzureOpenAIAgent

# Path to the actual MCP configurations, assuming it's in the project root
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")
# Ensure skill files are discoverable; they are expected in PROJECT_ROOT
# (e.g., realtime_processing_skill.py)

class TestAzureOpenAIAgentMCP(unittest.TestCase): # Renamed for clarity

    def setUp(self):
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        # Ensure a valid (even if minimal) mcp_configurations.json exists for agent init
        # The ACTUAL_MCP_CONFIG_FILE should be the full one.
        if not os.path.exists(ACTUAL_MCP_CONFIG_FILE):
            # Create a minimal one if the main one is missing, just for robust test setup
            # This should ideally not happen if files are version controlled.
            minimal_mcp_config = {"mcp_version": "1.0.0", "models": [], "skills": [], "context_types": []}
            with open(ACTUAL_MCP_CONFIG_FILE, 'w') as f:
                json.dump(minimal_mcp_config, f)
            self.created_minimal_mcp_config = True
        else:
            self.created_minimal_mcp_config = False


        self.patches = {
            'AzureOpenAI_client': patch('azure_openai_agent.AzureOpenAI'),
            'factual_memory_path': patch('azure_openai_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('azure_openai_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            # Agent now loads MCP_CONFIG_FILE_PATH directly, no need to patch its internal var if agent uses the global
            # But if agent has MCP_CONFIG_FILE_PATH as a global constant that it uses, patching that might be needed for specific test scenarios.
            # The current agent loads "mcp_configurations.json" directly.
        }

        self.mock_azure_openai_client_constructor = self.patches['AzureOpenAI_client'].start()
        self.mock_azure_client_instance = MagicMock()
        self.mock_azure_openai_client_constructor.return_value = self.mock_azure_client_instance

        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()

        # Agent config is now general, not MCP specific skill configs
        self.agent = AzureOpenAIAgent(
            azure_api_key="dummy_key",
            azure_endpoint="dummy_endpoint",
            llm_deployment_name="dummy_deployment"
        )

    def tearDown(self):
        for patcher in self.patches.values():
            patcher.stop()
        os.remove(self.temp_factual_mem_file.name)
        os.remove(self.temp_procedural_mem_file.name)
        if self.created_minimal_mcp_config and os.path.exists(ACTUAL_MCP_CONFIG_FILE):
             # Clean up minimal config if it was created by setup, to not affect other processes
             # Ideally, tests should use a fixture copy of mcp_configurations.json
             # For now, this cleanup is basic.
             # os.remove(ACTUAL_MCP_CONFIG_FILE)
             pass # Decided against removing it to avoid side effects if it was genuinely there.


    def test_mcp_registry_loading(self):
        """Test that the agent loads and processes MCP configurations correctly."""
        self.assertIsNotNone(self.agent.mcp_registry)
        self.assertIn("mcp_version", self.agent.mcp_registry)
        self.assertIn("models", self.agent.mcp_registry)
        self.assertIn("skills", self.agent.mcp_registry)
        self.assertIn("context_types", self.agent.mcp_registry)
        self.assertIn("skills_map", self.agent.mcp_registry)
        self.assertTrue(len(self.agent.mcp_registry["skills"]) > 0, "Skills should be loaded")
        self.assertTrue(len(self.agent.mcp_registry["skills_map"]) > 0, "Skills map should be populated")
        # Check if a known skill ID is in the map
        self.assertIn("realtime_data_processor", self.agent.mcp_registry["skills_map"])


    def _mock_llm_chat_response(self, response_content_json_str):
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = response_content_json_str
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [mock_choice]
        self.mock_azure_client_instance.chat.completions.create.return_value = mock_completion_response

    def test_llm_determines_skill_id(self):
        """Test LLM response parsing for skill_id and parameters."""
        mock_llm_output = {
            "intent": "Process data with a skill",
            "skill_id": "realtime_data_processor", # Skill ID from mcp_configurations.json
            "parameters": {"raw_data_json": "[{\"id\":\"test\"}]"},
            "requires_further_clarification": False,
            "clarification_question": "None",
            "response_to_user": "Processing now."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_output))

        intent_data = self.agent._determine_intent_and_params_with_llm("some query")
        self.assertEqual(intent_data["skill_id"], "realtime_data_processor")
        self.assertEqual(intent_data["parameters"]["raw_data_json"], "[{\"id\":\"test\"}]")

    def test_prepare_skill_context_memory_snapshots(self):
        """Test _prepare_skill_context correctly includes memory snapshots."""
        self.agent.add_factual_memory("user_name", "test_user")
        self.agent.add_procedural_memory("test_task", ["step1"])

        # Mock skill definition that expects memory snapshots
        mock_skill_def = {
            "id": "test_skill_with_mem",
            "expected_context": ["factual_memory_snapshot", "procedural_memory_snapshot", "user_query_context"]
            # user_query_context is not sourced from agent memory in current _prepare_skill_context
        }
        with patch.dict(self.agent.mcp_registry['skills_map'], {'test_skill_with_mem': mock_skill_def}):
            context_data = self.agent._prepare_skill_context("test_skill_with_mem")

        self.assertIn("factual_memory_snapshot", context_data)
        self.assertEqual(context_data["factual_memory_snapshot"]["user_name"], "test_user")
        self.assertIn("procedural_memory_snapshot", context_data)
        self.assertEqual(context_data["procedural_memory_snapshot"]["test_task"]["steps"], ["step1"])
        self.assertIn("agent_name", context_data) # General context also added


    @patch('azure_openai_agent.AzureOpenAIAgent.invoke_mcp_skill') # High-level mock for this test
    def test_process_query_uses_skill(self, mock_invoke_skill):
        """Test process_query flow when a skill is identified and invoked."""
        mock_llm_output = {
            "intent": "Use a skill",
            "skill_id": "realtime_data_processor",
            "parameters": {"param1": "value1"},
            "requires_further_clarification": False,
            "response_to_user": "Will use skill."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_output))

        mock_skill_response = {"status": "success", "data": {"result": "skill_done"}}
        mock_invoke_skill.return_value = mock_skill_response

        with patch.object(self.agent, '_prepare_skill_context', return_value={"prepared": True}) as mock_prepare_context:
            response = self.agent.process_query("test query for skill")

        mock_prepare_context.assert_called_once_with("realtime_data_processor")
        mock_invoke_skill.assert_called_once_with(
            "realtime_data_processor",
            {"param1": "value1"},
            {"prepared": True}
        )
        self.assertIn("Skill 'realtime_data_processor' executed successfully.", response)
        self.assertIn(json.dumps({"result": "skill_done"}), response)


    @patch('importlib.util') # To mock dynamic importing
    @patch.dict(sys.modules, {}) # Ensure modules are freshly "imported" for the test
    def test_invoke_mcp_skill_successful_invocation(self, mock_importlib_util):
        """Test invoke_mcp_skill's dynamic import and call mechanism."""
        skill_id = "realtime_data_processor" # Assumes this skill is in mcp_registry

        # Mock skill module and handler
        mock_skill_module = MagicMock()
        mock_skill_class_constructor = MagicMock() # Mock for RealTimeStreamingMCP class
        mock_skill_instance = MagicMock() # Mock for instance of RealTimeStreamingMCP
        mock_skill_method = MagicMock(return_value={"status": "success", "data": "mock skill executed"}) # Mock for process_stream_data method

        # Setup importlib mocking
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        mock_importlib_util.module_from_spec.return_value = mock_skill_module

        # Configure the mocked module to have the class, and class to have the method
        # The handler_path is "RealTimeStreamingMCP.process_stream_data"
        setattr(mock_skill_module, "RealTimeStreamingMCP", mock_skill_class_constructor)
        mock_skill_class_constructor.return_value = mock_skill_instance
        setattr(mock_skill_instance, "process_stream_data", mock_skill_method)

        parameters = {"raw_data_json": "test_data"}
        context = {"user_id": "tester"}

        # Ensure the skill definition exists in the agent's registry for this test
        # (it should if ACTUAL_MCP_CONFIG_FILE is loaded correctly by setUp)
        self.assertTrue(skill_id in self.agent.mcp_registry['skills_map'], f"{skill_id} not in skills_map")

        skill_response = self.agent.invoke_mcp_skill(skill_id, parameters, context)

        mock_importlib_util.spec_from_file_location.assert_called_once_with(
            self.agent.mcp_registry['skills_map'][skill_id]['handler_module'],
            f"{self.agent.mcp_registry['skills_map'][skill_id]['handler_module']}.py"
        )
        mock_spec.loader.exec_module.assert_called_once_with(mock_skill_module)
        mock_skill_class_constructor.assert_called_once_with() # Assuming constructor takes no args from agent
        mock_skill_method.assert_called_once_with(parameters, context)
        self.assertEqual(skill_response["status"], "success")
        self.assertEqual(skill_response["data"], "mock skill executed")

    def test_invoke_mcp_skill_module_not_found(self):
        skill_id = "non_existent_skill_module"
        # Add a dummy skill def for this test case to agent's registry
        self.agent.mcp_registry['skills_map'][skill_id] = {
            "id": skill_id, "name": "Test Non Existent Module",
            "handler_module": "imaginary_module",
            "handler_class_or_function": "SomeHandler.do_work"
        }

        # Make importlib spec_from_file_location return None to simulate module not found by spec system
        with patch('importlib.util.spec_from_file_location', return_value=None):
            response = self.agent.invoke_mcp_skill(skill_id, {}, {})
        self.assertEqual(response["status"], "error")
        self.assertIn("Could not create module spec for imaginary_module", response["error_message"])

    def test_invoke_mcp_skill_file_not_found_os_error(self):
        skill_id = "skill_file_not_found"
        self.agent.mcp_registry['skills_map'][skill_id] = {
            "id": skill_id, "name": "Test File Not Found",
            "handler_module": "truly_missing_module_file", # This .py file won't exist
            "handler_class_or_function": "Handler.method"
        }
        # importlib.util.spec_from_file_location will raise FileNotFoundError if the .py file is missing
        # This is caught by the general Exception in invoke_mcp_skill if not caught earlier by spec returning None.
        # Let's refine this to mock the FileNotFoundError directly if spec_from_file_location itself would raise it.
        # For this test, we assume spec_from_file_location itself might fail if the file is not there.
        # The agent's current invoke_mcp_skill catches FileNotFoundError specifically.
        with patch('importlib.util.spec_from_file_location', side_effect=FileNotFoundError("Simulated FileNotFoundError")):
             response = self.agent.invoke_mcp_skill(skill_id, {}, {})

        # Based on current agent code, FileNotFoundError is caught by the generic Exception handler.
        # To specifically test the FileNotFoundError block in `invoke_mcp_skill`, that block would need to be reachable
        # before the spec_from_file_location call, which is not the case.
        # The current FileNotFoundError catch in invoke_mcp_skill is if the *module_name*.py file isn't found.
        # This is what the above mock simulates.

        # The agent's code is:
        # except FileNotFoundError:
        #    logging.error(f"Skill module file not found: {skill_module_path_py}")
        #    return {"status": "error", "error_message": f"Skill module '{module_name}.py' not found."}
        # This is hit if spec_from_file_location itself raises FileNotFoundError.
        self.assertEqual(response["status"], "error")
        # The error message comes from the FileNotFoundError catch block if spec_from_file_location raises it.
        # If spec_from_file_location does *not* raise FileNotFoundError but returns None, then the ImportError path is taken.
        # The current mock will lead to the "Skill module ... not found" if FileNotFoundError is raised by spec_from_file_location.
        # If spec_from_file_location works but the module can't be loaded by loader, it's an ImportError.
        # The current test setup makes spec_from_file_location raise FileNotFoundError.
        self.assertIn(f"Skill module 'truly_missing_module_file.py' not found.", response["error_message"])


    # Memory tests (load/save) are largely similar to previous agent versions,
    # ensuring they still work with the new init.
    def test_initialization_loads_existing_memory(self):
        factual_data = {"fact_azure_mcp": "val_azure_mcp"}
        procedural_data = {"proc_azure_mcp": {"steps": ["step_mcp"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        # Re-initialize agent
        reloaded_agent = AzureOpenAIAgent(azure_api_key="dummy_key", azure_endpoint="dummy_endpoint")
        self.assertEqual(reloaded_agent.factual_memory, factual_data)
        self.assertEqual(reloaded_agent.procedural_memory["proc_azure_mcp"]["steps"], procedural_data["proc_azure_mcp"]["steps"])

if __name__ == '__main__':
    unittest.main()

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
from azure_openai_agent import AzureOpenAIAgent

# Path to the actual MCP configurations, assuming it's in the project root
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestAzureOpenAIAgent(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        # Create temporary files for memory
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        
        # Close them immediately so the agent can open them if needed, or so we can write to them.
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        # Patch the file paths used by the agent BEFORE agent initialization
        self.patches = {
            'AzureOpenAI_client': patch('azure_openai_agent.AzureOpenAI'),
            'factual_memory_path': patch('azure_openai_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('azure_openai_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            'mcp_config_path': patch('azure_openai_agent.MCP_CONFIG_FILE', ACTUAL_MCP_CONFIG_FILE),
            # We also need to ensure custom MCPs can be loaded.
            # The agent loads them using importlib.util.spec_from_file_location relative to its own file.
            # If custom_mcp_*.py files are in PROJECT_ROOT, this should work if tests are run from PROJECT_ROOT
            # or if azure_openai_agent.py correctly resolves paths (which it does by assuming they are co-located).
        }

        self.mock_azure_openai_client_constructor = self.patches['AzureOpenAI_client'].start()
        self.mock_azure_client_instance = MagicMock()
        self.mock_azure_openai_client_constructor.return_value = self.mock_azure_client_instance
        
        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()
        self.patches['mcp_config_path'].start()

        # Dummy agent config for MCPs
        self.agent_mcp_configs = {
            "streaming_mcp_config": {'validation_rules': {'value': {'type': float}}},
            "federated_mcp_config": {"global_model_id": "azure_test_model"},
            "robust_mcp_config": {'max_retries': 1}
        }

        self.agent = AzureOpenAIAgent(
            azure_api_key="dummy_key",
            azure_endpoint="dummy_endpoint",
            llm_deployment_name="dummy_deployment",
            agent_config=self.agent_mcp_configs
        )
        # Ensure custom MCPs are loaded (at least their names)
        self.assertTrue(len(self.agent.custom_mcps) > 0, "Custom MCPs were not loaded")


    def tearDown(self):
        """Clean up after each test."""
        for patcher in self.patches.values():
            patcher.stop()
        os.remove(self.temp_factual_mem_file.name)
        os.remove(self.temp_procedural_mem_file.name)

    def test_initialization_and_empty_memory_load(self):
        """Test agent initializes and loads empty memory if files are empty/new."""
        self.assertIsNotNone(self.agent.openai_client)
        self.assertEqual(self.agent.factual_memory, {})
        self.assertEqual(self.agent.procedural_memory, {})
        self.assertTrue(len(self.agent.mcp_configurations) > 0) # Loaded from actual file

    def test_memory_load_existing(self):
        """Test loading pre-populated memory files."""
        # Write dummy data to temp memory files BEFORE agent re-initialization
        factual_data = {"fact1": "value1"}
        procedural_data = {"proc1": {"steps": ["stepA"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f:
            json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f:
            json.dump(procedural_data, f)

        # Re-initialize agent to load these files
        agent_reloaded = AzureOpenAIAgent(
            azure_api_key="dummy_key", azure_endpoint="dummy_endpoint", agent_config=self.agent_mcp_configs
        )
        self.assertEqual(agent_reloaded.factual_memory, factual_data)
        self.assertEqual(agent_reloaded.procedural_memory["proc1"]["steps"], procedural_data["proc1"]["steps"])


    def test_factual_memory_operations(self):
        """Test adding, getting, and saving factual memory."""
        self.agent.add_factual_memory("test_fact_key", "test_fact_value")
        self.assertEqual(self.agent.get_factual_memory("test_fact_key"), "test_fact_value")
        # Verify it was saved
        with open(self.temp_factual_mem_file.name, 'r') as f:
            saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_fact_key"], "test_fact_value")

    def test_procedural_memory_operations(self):
        """Test adding, getting, and saving procedural memory."""
        self.agent.add_procedural_memory("test_proc_task", ["step1", "step2"])
        retrieved_proc = self.agent.get_procedural_memory("test_proc_task")
        self.assertEqual(retrieved_proc["steps"], ["step1", "step2"])
        # Verify it was saved
        with open(self.temp_procedural_mem_file.name, 'r') as f:
            saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_proc_task"]["steps"], ["step1", "step2"])

    def _mock_llm_chat_response(self, response_content_json_str):
        """Helper to mock the Azure OpenAI chat completion response."""
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = response_content_json_str
        
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [mock_choice]
        self.mock_azure_client_instance.chat.completions.create.return_value = mock_completion_response

    def test_llm_intent_parsing(self):
        """Test if the agent correctly parses the LLM's JSON response for intent."""
        mock_llm_json_output = {
            "intent": "Test Intent",
            "mcp_tool_name": "RealTimeStreamingMCP",
            "mcp_config_name": None,
            "parameters": {"raw_data_json": "[{\"test\":\"data\"}]", "method_name": "process_stream_data"},
            "requires_further_clarification": False,
            "clarification_question": None,
            "response_to_user": "Processing your test data."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))
        
        # We are testing the parsing part, which is internal to process_query
        # but drives its behavior.
        # We can also call _determine_intent_and_params_with_llm directly for more focused test.
        parsed_intent_data = self.agent._determine_intent_and_params_with_llm("some query")
        
        self.assertEqual(parsed_intent_data["intent"], "Test Intent")
        self.assertEqual(parsed_intent_data["mcp_tool_name"], "RealTimeStreamingMCP")
        self.assertEqual(parsed_intent_data["parameters"]["raw_data_json"], "[{\"test\":\"data\"}]")

    @patch('custom_mcp_1.RealTimeStreamingMCP.process_stream_data') # Patch the actual MCP method
    def test_tool_usage_streaming_mcp(self, mock_streaming_process_method):
        """Test that the agent calls the correct MCP method based on LLM intent."""
        mock_llm_json_output = {
            "intent": "Process streaming data",
            "mcp_tool_name": "RealTimeStreamingMCP",
            "mcp_config_name": None,
            "parameters": {"raw_data_json": "[{\"id\":\"s1\"}]", "method_name": "process_stream_data"},
            "requires_further_clarification": False,
            "response_to_user": "Using RealTimeStreamingMCP."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))
        mock_streaming_process_method.return_value = "Streaming MCP processed data"

        response = self.agent.process_query("process my stream data '[{\"id\":\"s1\"}]'")
        
        mock_streaming_process_method.assert_called_once_with(raw_data_json="[{\"id\":\"s1\"}]")
        self.assertIn("Used RealTimeStreamingMCP: Streaming MCP processed data", response)


    def test_llm_api_error_handling(self):
        """Test agent's behavior when LLM call fails."""
        self.mock_azure_client_instance.chat.completions.create.side_effect = Exception("Azure API Error")
        
        response = self.agent.process_query("a query that will cause an error")
        self.assertIn("Error communicating with Azure OpenAI (Chat): Azure API Error", response)
        # Check if the error was recorded in procedural memory (as per current agent logic for general queries)
        # This depends on where the error is caught and how it's handled.
        # The current _determine_intent_and_params_with_llm returns a fallback JSON.
        # The process_query will then use that fallback.
        self.assertIn("I'm having a little trouble understanding. Could you try asking in a different way?", response)


    def test_mcp_config_selection_flow(self):
        """Test selecting an MCP configuration by name."""
        mock_llm_json_output = {
            "intent": "Inquire about MCP config",
            "mcp_tool_name": None,
            "mcp_config_name": "Standard HTTPS/JSON MCP", # Assuming this exists in mcp_configurations.json
            "parameters": {},
            "requires_further_clarification": False,
            "response_to_user": "Here is info about Standard HTTPS/JSON MCP." # LLM might provide this
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))

        response = self.agent.process_query("Tell me about the Standard HTTPS/JSON MCP")
        self.assertIn("Found MCP configuration: 'Standard HTTPS/JSON MCP'", response)
        self.assertIn("HTTPS", response) # Check if some detail is present
        self.assertIsNotNone(self.agent.get_factual_memory("last_selected_mcp_config"))
        self.assertEqual(self.agent.get_factual_memory("last_selected_mcp_config")["name"], "Standard HTTPS/JSON MCP")

    def test_clarification_flow(self):
        """Test the clarification flow when LLM requires more info."""
        clarification_q = "What data do you want to stream?"
        mock_llm_json_output = {
            "intent": "Stream data",
            "mcp_tool_name": "RealTimeStreamingMCP",
            "parameters": {},
            "requires_further_clarification": True,
            "clarification_question": clarification_q,
            "response_to_user": clarification_q # Often the same for this case
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))

        query = "I want to stream some data."
        response = self.agent.process_query(query)

        self.assertEqual(response, clarification_q)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_for_query"), query)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_question"), clarification_q)


if __name__ == '__main__':
    unittest.main()

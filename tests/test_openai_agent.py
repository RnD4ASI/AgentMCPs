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
from openai_agent import OpenAIAgent # Changed import

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestOpenAIAgent(unittest.TestCase): # Changed class name

    def setUp(self):
        """Set up for each test."""
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        self.patches = {
            # Patched the standard OpenAI client
            'OpenAI_client': patch('openai_agent.OpenAI'), 
            'factual_memory_path': patch('openai_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('openai_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            'mcp_config_path': patch('openai_agent.MCP_CONFIG_FILE', ACTUAL_MCP_CONFIG_FILE),
        }

        self.mock_openai_client_constructor = self.patches['OpenAI_client'].start()
        self.mock_openai_client_instance = MagicMock()
        self.mock_openai_client_constructor.return_value = self.mock_openai_client_instance
        
        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()
        self.patches['mcp_config_path'].start()

        self.agent_mcp_configs = {
            "streaming_mcp_config": {'validation_rules': {'value': {'type': float}}},
            "federated_mcp_config": {"global_model_id": "openai_test_model"}, # Differentiation
            "robust_mcp_config": {'max_retries': 1}
        }

        self.agent = OpenAIAgent( # Changed agent instantiation
            api_key="dummy_openai_key", # Changed param name
            model_name="gpt-3.5-turbo", # Standard model name
            agent_config=self.agent_mcp_configs
        )
        self.assertTrue(len(self.agent.custom_mcps) > 0, "Custom MCPs were not loaded")


    def tearDown(self):
        """Clean up after each test."""
        for patcher in self.patches.values():
            patcher.stop()
        os.remove(self.temp_factual_mem_file.name)
        os.remove(self.temp_procedural_mem_file.name)

    def test_initialization_and_empty_memory_load(self):
        self.assertIsNotNone(self.agent.openai_client)
        self.assertEqual(self.agent.factual_memory, {})
        self.assertEqual(self.agent.procedural_memory, {})
        self.assertTrue(len(self.agent.mcp_configurations) > 0)

    def test_memory_load_existing(self):
        factual_data = {"fact_openai": "value_openai"} # Differentiation
        procedural_data = {"proc_openai": {"steps": ["step_openai"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        agent_reloaded = OpenAIAgent( # Changed agent
            api_key="dummy_openai_key", model_name="gpt-3.5-turbo", agent_config=self.agent_mcp_configs
        )
        self.assertEqual(agent_reloaded.factual_memory, factual_data)
        self.assertEqual(agent_reloaded.procedural_memory["proc_openai"]["steps"], procedural_data["proc_openai"]["steps"])


    def test_factual_memory_operations(self):
        self.agent.add_factual_memory("test_fact_openai_key", "test_fact_openai_value")
        self.assertEqual(self.agent.get_factual_memory("test_fact_openai_key"), "test_fact_openai_value")
        with open(self.temp_factual_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_fact_openai_key"], "test_fact_openai_value")

    def test_procedural_memory_operations(self):
        self.agent.add_procedural_memory("test_proc_openai_task", ["step_A", "step_B"])
        retrieved_proc = self.agent.get_procedural_memory("test_proc_openai_task")
        self.assertEqual(retrieved_proc["steps"], ["step_A", "step_B"])
        with open(self.temp_procedural_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_proc_openai_task"]["steps"], ["step_A", "step_B"])

    def _mock_llm_chat_response(self, response_content_json_str):
        """Helper to mock the OpenAI chat completion response."""
        # Structure is same as Azure OpenAI's for chat completions
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = response_content_json_str
        
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [mock_choice]
        # Path to the mocked method is on the *instance* of the client
        self.mock_openai_client_instance.chat.completions.create.return_value = mock_completion_response


    def test_llm_intent_parsing(self):
        mock_llm_json_output = {
            "intent": "OpenAI Test Intent", # Differentiation
            "mcp_tool_name": "RobustMCP", # Different MCP for variety
            "mcp_config_name": None,
            "parameters": {"message_json": "{\"id\":\"msg1\"}", "method_name": "process_message"},
            "requires_further_clarification": False,
            "clarification_question": None,
            "response_to_user": "Processing your message with RobustMCP."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))
        
        parsed_intent_data = self.agent._determine_intent_and_params_with_llm("some query for openai agent")
        
        self.assertEqual(parsed_intent_data["intent"], "OpenAI Test Intent")
        self.assertEqual(parsed_intent_data["mcp_tool_name"], "RobustMCP")
        self.assertEqual(parsed_intent_data["parameters"]["message_json"], "{\"id\":\"msg1\"}")

    @patch('custom_mcp_3.RobustMCP.process_message') # Patching RobustMCP method
    def test_tool_usage_robust_mcp(self, mock_robust_process_method):
        mock_llm_json_output = {
            "intent": "Process a robust message",
            "mcp_tool_name": "RobustMCP",
            "mcp_config_name": None,
            "parameters": {"message_json": "{\"id\":\"msgRob\"}", "method_name": "process_message"},
            "requires_further_clarification": False,
            "response_to_user": "Using RobustMCP for your message."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))
        mock_robust_process_method.return_value = "Robust MCP processed message"

        response = self.agent.process_query("process this robust message '{\"id\":\"msgRob\"}'")
        
        mock_robust_process_method.assert_called_once_with(message_json="{\"id\":\"msgRob\"}")
        self.assertIn("Used RobustMCP: Robust MCP processed message", response)


    def test_llm_api_error_handling(self):
        self.mock_openai_client_instance.chat.completions.create.side_effect = Exception("OpenAI API Error")
        
        response = self.agent.process_query("a query that triggers an API error")
        # The agent's _get_openai_chat_completion catches the generic Exception
        # and returns a string "Error communicating with OpenAI: OpenAI API Error"
        # This string is then used in _determine_intent_and_params_with_llm to create a fallback JSON.
        self.assertIn("I'm having a little trouble understanding. Could you try asking in a different way?", response)


    def test_mcp_config_selection_flow(self):
        # Assuming "High-Performance gRPC MCP" exists in mcp_configurations.json
        config_name_to_test = "High-Performance gRPC MCP" 
        mock_llm_json_output = {
            "intent": "Inquire about gRPC MCP config",
            "mcp_tool_name": None,
            "mcp_config_name": config_name_to_test,
            "parameters": {},
            "requires_further_clarification": False,
            "response_to_user": f"Info about {config_name_to_test}."
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))

        response = self.agent.process_query(f"Tell me about {config_name_to_test}")
        self.assertIn(f"Found MCP configuration: '{config_name_to_test}'", response)
        self.assertIn("gRPC", response) 
        self.assertIsNotNone(self.agent.get_factual_memory("last_selected_mcp_config"))
        self.assertEqual(self.agent.get_factual_memory("last_selected_mcp_config")["name"], config_name_to_test)

    def test_clarification_flow(self):
        clarification_q = "Which model do you want to update for federated learning?"
        mock_llm_json_output = {
            "intent": "Federated learning update",
            "mcp_tool_name": "FederatedLearningMCP",
            "parameters": {"method_name": "submit_model_update"}, # Missing other params
            "requires_further_clarification": True,
            "clarification_question": clarification_q,
            "response_to_user": clarification_q
        }
        self._mock_llm_chat_response(json.dumps(mock_llm_json_output))

        query = "I want to submit a federated learning update."
        response = self.agent.process_query(query)

        self.assertEqual(response, clarification_q)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_for_query"), query)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_question"), clarification_q)


if __name__ == '__main__':
    unittest.main()

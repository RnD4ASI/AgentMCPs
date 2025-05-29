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
from anthropic_agent import AnthropicAgent # Changed import

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestAnthropicAgent(unittest.TestCase): # Changed class name

    def setUp(self):
        """Set up for each test."""
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        self.patches = {
            'Anthropic_client': patch('anthropic_agent.Anthropic'), # Patched Anthropic client
            'factual_memory_path': patch('anthropic_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('anthropic_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            'mcp_config_path': patch('anthropic_agent.MCP_CONFIG_FILE', ACTUAL_MCP_CONFIG_FILE),
        }

        self.mock_anthropic_client_constructor = self.patches['Anthropic_client'].start()
        self.mock_anthropic_client_instance = MagicMock()
        self.mock_anthropic_client_constructor.return_value = self.mock_anthropic_client_instance
        
        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()
        self.patches['mcp_config_path'].start()

        self.agent_mcp_configs = {
            "streaming_mcp_config": {'validation_rules': {'temp': {'type': float}}},
            "federated_mcp_config": {"global_model_id": "anthropic_test_model"}, # Differentiation
            "robust_mcp_config": {'max_retries': 2}
        }

        self.agent = AnthropicAgent( # Changed agent instantiation
            api_key="dummy_anthropic_key", # Changed param name
            model_name="claude-3-haiku-20240307", # Example model
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
        self.assertIsNotNone(self.agent.anthropic_client)
        self.assertEqual(self.agent.factual_memory, {})
        self.assertEqual(self.agent.procedural_memory, {})
        self.assertTrue(len(self.agent.mcp_configurations) > 0)

    def test_memory_load_existing(self):
        factual_data = {"fact_anthropic": "value_anthropic"} # Differentiation
        procedural_data = {"proc_anthropic": {"steps": ["step_anthropic"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        agent_reloaded = AnthropicAgent( # Changed agent
            api_key="dummy_anthropic_key", model_name="claude-3-haiku-20240307", agent_config=self.agent_mcp_configs
        )
        self.assertEqual(agent_reloaded.factual_memory, factual_data)
        self.assertEqual(agent_reloaded.procedural_memory["proc_anthropic"]["steps"], procedural_data["proc_anthropic"]["steps"])


    def test_factual_memory_operations(self):
        self.agent.add_factual_memory("test_fact_anthropic_key", "test_fact_anthropic_value")
        self.assertEqual(self.agent.get_factual_memory("test_fact_anthropic_key"), "test_fact_anthropic_value")
        with open(self.temp_factual_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_fact_anthropic_key"], "test_fact_anthropic_value")

    def test_procedural_memory_operations(self):
        self.agent.add_procedural_memory("test_proc_anthropic_task", ["step_X", "step_Y"])
        retrieved_proc = self.agent.get_procedural_memory("test_proc_anthropic_task")
        self.assertEqual(retrieved_proc["steps"], ["step_X", "step_Y"])
        with open(self.temp_procedural_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_proc_anthropic_task"]["steps"], ["step_X", "step_Y"])

    def _mock_llm_completion_response(self, response_text_str):
        """Helper to mock the Anthropic messages.create response."""
        # Anthropic's response structure: response.content is a list of blocks.
        # We're interested in the text from the first TextBlock.
        mock_text_block = MagicMock()
        mock_text_block.text = response_text_str 
        # mock_text_block.type = "text" # if needed to be very specific

        mock_completion_response = MagicMock()
        mock_completion_response.content = [mock_text_block]
        # mock_completion_response.model = self.agent.model_name
        # mock_completion_response.stop_reason = "end_turn"
        
        self.mock_anthropic_client_instance.messages.create.return_value = mock_completion_response


    def test_llm_intent_parsing(self):
        mock_llm_json_output = {
            "intent": "Anthropic Test Intent", # Differentiation
            "mcp_tool_name": "FederatedLearningMCP", # Different MCP
            "mcp_config_name": None,
            "parameters": {"round_id": 1, "method_name": "start_new_round"},
            "requires_further_clarification": False,
            "clarification_question": None,
            "response_to_user": "Starting new FL round with Anthropic."
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_json_output))
        
        parsed_intent_data = self.agent._determine_intent_and_params_with_llm("some query for anthropic agent")
        
        self.assertEqual(parsed_intent_data["intent"], "Anthropic Test Intent")
        self.assertEqual(parsed_intent_data["mcp_tool_name"], "FederatedLearningMCP")
        self.assertEqual(parsed_intent_data["parameters"]["round_id"], 1)

    @patch('custom_mcp_2.FederatedLearningMCP.start_new_round') # Patching FL MCP method
    def test_tool_usage_federated_mcp(self, mock_fl_start_round_method):
        mock_llm_json_output = {
            "intent": "Start FL round",
            "mcp_tool_name": "FederatedLearningMCP",
            "mcp_config_name": None,
            "parameters": {"round_id": 5, "method_name": "start_new_round"},
            "requires_further_clarification": False,
            "response_to_user": "Using FederatedLearningMCP to start round 5."
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_json_output))
        # Assume start_new_round returns the model parameters or some status
        mock_fl_start_round_method.return_value = {"status": "Round 5 started", "params": {}} 

        response = self.agent.process_query("start federated learning round 5")
        
        mock_fl_start_round_method.assert_called_once_with(round_id=5)
        self.assertIn("Used FederatedLearningMCP: {'status': 'Round 5 started'", response)


    def test_llm_api_error_handling(self):
        self.mock_anthropic_client_instance.messages.create.side_effect = Exception("Anthropic API Error")
        
        response = self.agent.process_query("a query causing anthropic api error")
        # Based on current agent implementation for Anthropic
        self.assertIn("I'm having a little difficulty understanding your request. Could you try phrasing it differently?", response)


    def test_mcp_config_selection_flow(self):
        # Assuming "Legacy SOAP/XML MCP" exists
        config_name_to_test = "Legacy SOAP/XML MCP" 
        mock_llm_json_output = {
            "intent": "Inquire about SOAP MCP config",
            "mcp_tool_name": None,
            "mcp_config_name": config_name_to_test,
            "parameters": {},
            "requires_further_clarification": False,
            "response_to_user": f"Info about {config_name_to_test}."
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_json_output))

        response = self.agent.process_query(f"Tell me about the {config_name_to_test}")
        self.assertIn(f"Found MCP configuration: '{config_name_to_test}'", response)
        self.assertIn("SOAP", response) 
        self.assertIsNotNone(self.agent.get_factual_memory("last_selected_mcp_config"))
        self.assertEqual(self.agent.get_factual_memory("last_selected_mcp_config")["name"], config_name_to_test)

    def test_clarification_flow(self):
        clarification_q = "What is the payload for the robust message?"
        mock_llm_json_output = {
            "intent": "Send robust message",
            "mcp_tool_name": "RobustMCP",
            "parameters": {"method_name": "process_message"}, # Missing message_json
            "requires_further_clarification": True,
            "clarification_question": clarification_q,
            "response_to_user": clarification_q
        }
        self._mock_llm_completion_response(json.dumps(mock_llm_json_output))

        query = "I want to send a message robustly."
        response = self.agent.process_query(query)

        self.assertEqual(response, clarification_q)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_for_query"), query)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_question"), clarification_q)


if __name__ == '__main__':
    unittest.main()

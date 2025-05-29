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
from gemini_agent import GeminiAgent # Changed import
import google.generativeai as genai # For mocking response types if needed

# Path to the actual MCP configurations
ACTUAL_MCP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "mcp_configurations.json")

class TestGeminiAgent(unittest.TestCase): # Changed class name

    def setUp(self):
        """Set up for each test."""
        self.temp_factual_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_procedural_mem_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        self.temp_factual_mem_file.close()
        self.temp_procedural_mem_file.close()

        self.patches = {
            # Patch genai.configure and genai.GenerativeModel
            'genai_configure': patch('gemini_agent.genai.configure'),
            'genai_GenerativeModel': patch('gemini_agent.genai.GenerativeModel'),
            'factual_memory_path': patch('gemini_agent.FACTUAL_MEMORY_FILE', self.temp_factual_mem_file.name),
            'procedural_memory_path': patch('gemini_agent.PROCEDURAL_MEMORY_FILE', self.temp_procedural_mem_file.name),
            'mcp_config_path': patch('gemini_agent.MCP_CONFIG_FILE', ACTUAL_MCP_CONFIG_FILE),
        }

        self.mock_genai_configure = self.patches['genai_configure'].start()
        self.mock_genai_generativemodel_constructor = self.patches['genai_GenerativeModel'].start()
        
        self.mock_gemini_model_instance = MagicMock() # This is what GenerativeModel() returns
        self.mock_genai_generativemodel_constructor.return_value = self.mock_gemini_model_instance
        
        self.patches['factual_memory_path'].start()
        self.patches['procedural_memory_path'].start()
        self.patches['mcp_config_path'].start()

        self.agent_mcp_configs = {
            "streaming_mcp_config": {'validation_rules': {'data': {'type': str}}},
            "federated_mcp_config": {"global_model_id": "gemini_test_model"}, # Differentiation
            "robust_mcp_config": {'max_retries': 0} # No retries for faster test
        }

        self.agent = GeminiAgent( # Changed agent instantiation
            api_key="dummy_gemini_key", # Changed param name
            model_name="gemini-pro", 
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
        self.assertIsNotNone(self.agent.gemini_model)
        self.mock_genai_configure.assert_called_once_with(api_key="dummy_gemini_key")
        self.mock_genai_generativemodel_constructor.assert_called_once_with(model_name="gemini-pro")
        self.assertEqual(self.agent.factual_memory, {})
        self.assertEqual(self.agent.procedural_memory, {})
        self.assertTrue(len(self.agent.mcp_configurations) > 0)

    def test_memory_load_existing(self):
        factual_data = {"fact_gemini": "value_gemini"} # Differentiation
        procedural_data = {"proc_gemini": {"steps": ["step_gemini"]}}
        with open(self.temp_factual_mem_file.name, 'w') as f: json.dump(factual_data, f)
        with open(self.temp_procedural_mem_file.name, 'w') as f: json.dump(procedural_data, f)

        agent_reloaded = GeminiAgent( # Changed agent
            api_key="dummy_gemini_key", model_name="gemini-pro", agent_config=self.agent_mcp_configs
        )
        self.assertEqual(agent_reloaded.factual_memory, factual_data)
        self.assertEqual(agent_reloaded.procedural_memory["proc_gemini"]["steps"], procedural_data["proc_gemini"]["steps"])

    def test_factual_memory_operations(self):
        self.agent.add_factual_memory("test_fact_gemini_key", "test_fact_gemini_value")
        self.assertEqual(self.agent.get_factual_memory("test_fact_gemini_key"), "test_fact_gemini_value")
        with open(self.temp_factual_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_fact_gemini_key"], "test_fact_gemini_value")

    def test_procedural_memory_operations(self):
        self.agent.add_procedural_memory("test_proc_gemini_task", ["step_alpha", "step_beta"])
        retrieved_proc = self.agent.get_procedural_memory("test_proc_gemini_task")
        self.assertEqual(retrieved_proc["steps"], ["step_alpha", "step_beta"])
        with open(self.temp_procedural_mem_file.name, 'r') as f: saved_mem = json.load(f)
        self.assertEqual(saved_mem["test_proc_gemini_task"]["steps"], ["step_alpha", "step_beta"])

    def _mock_llm_generate_content_response(self, response_text_str):
        """Helper to mock the Gemini generate_content response."""
        mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
        # Simulate the structure Gemini API client returns
        mock_response.text = response_text_str
        # For safety checks:
        mock_response.candidates = [MagicMock()] # Ensure candidates list is not empty
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = None
        mock_response.prompt_feedback.safety_ratings = []
        
        self.mock_gemini_model_instance.generate_content.return_value = mock_response

    def test_llm_intent_parsing(self):
        mock_llm_json_output = {
            "intent": "Gemini Test Intent", # Differentiation
            "mcp_tool_name": "RealTimeStreamingMCP", 
            "mcp_config_name": None,
            "parameters": {"raw_data_json": "[{\"gemini_data\":true}]", "method_name": "process_stream_data"},
            "requires_further_clarification": False,
            "clarification_question": None,
            "response_to_user": "Processing your data with Gemini and StreamingMCP."
        }
        # Gemini agent expects the JSON string directly, without markdown.
        self._mock_llm_generate_content_response(json.dumps(mock_llm_json_output))
        
        parsed_intent_data = self.agent._determine_intent_and_params_with_llm("some query for gemini agent")
        
        self.assertEqual(parsed_intent_data["intent"], "Gemini Test Intent")
        self.assertEqual(parsed_intent_data["mcp_tool_name"], "RealTimeStreamingMCP")
        self.assertEqual(parsed_intent_data["parameters"]["raw_data_json"], "[{\"gemini_data\":true}]")

    @patch('custom_mcp_1.RealTimeStreamingMCP.process_stream_data') # Patching Streaming MCP
    def test_tool_usage_streaming_mcp_with_gemini(self, mock_streaming_process_method):
        mock_llm_json_output = {
            "intent": "Process stream with Gemini",
            "mcp_tool_name": "RealTimeStreamingMCP",
            "mcp_config_name": None,
            "parameters": {"raw_data_json": "[{\"stream_item\":1}]", "method_name": "process_stream_data"},
            "requires_further_clarification": False,
            "response_to_user": "Gemini using RealTimeStreamingMCP."
        }
        self._mock_llm_generate_content_response(json.dumps(mock_llm_json_output))
        mock_streaming_process_method.return_value = "Gemini stream processed"

        response = self.agent.process_query("process this stream '[{\"stream_item\":1}]'")
        
        mock_streaming_process_method.assert_called_once_with(raw_data_json="[{\"stream_item\":1}]")
        self.assertIn("Used RealTimeStreamingMCP: Gemini stream processed", response)

    def test_llm_api_error_handling(self):
        self.mock_gemini_model_instance.generate_content.side_effect = Exception("Gemini API Error")
        
        response = self.agent.process_query("a query causing gemini api error")
        # Based on current agent implementation for Gemini
        self.assertIn("I'm having a little difficulty understanding your request with Gemini. Could you try phrasing it differently?", response)

    def test_llm_api_blocked_prompt(self):
        """Test handling of a blocked prompt by Gemini API."""
        mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response.text = None # No text if blocked
        mock_response.candidates = [] # Or None, depending on actual API behavior for blocks
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_response.prompt_feedback.safety_ratings = [MagicMock(category="HARM_CATEGORY_HARASSMENT", probability="HIGH")]
        
        self.mock_gemini_model_instance.generate_content.return_value = mock_response
        
        llm_response_str = self.agent._get_gemini_completion("a problematic prompt")
        self.assertIn("Error: Gemini API call blocked due to 'SAFETY'", llm_response_str)
        self.assertIn("HARM_CATEGORY_HARASSMENT: HIGH", llm_response_str)

        # Now test the full query processing flow
        response = self.agent.process_query("a problematic prompt")
        self.assertIn("I'm having a little difficulty understanding your request with Gemini. Could you try phrasing it differently?", response)


    def test_mcp_config_selection_flow(self):
        # Assuming "Simple HTTP/CSV MCP for Batch Data" exists
        config_name_to_test = "Simple HTTP/CSV MCP for Batch Data" 
        mock_llm_json_output = {
            "intent": "Inquire about CSV MCP config",
            "mcp_tool_name": None,
            "mcp_config_name": config_name_to_test,
            "parameters": {},
            "requires_further_clarification": False,
            "response_to_user": f"Info about {config_name_to_test}."
        }
        self._mock_llm_generate_content_response(json.dumps(mock_llm_json_output))

        response = self.agent.process_query(f"Tell me about the {config_name_to_test}")
        self.assertIn(f"Found MCP configuration: '{config_name_to_test}'", response)
        self.assertIn("HTTP", response) 
        self.assertIsNotNone(self.agent.get_factual_memory("last_selected_mcp_config"))
        self.assertEqual(self.agent.get_factual_memory("last_selected_mcp_config")["name"], config_name_to_test)

    def test_clarification_flow(self):
        clarification_q = "What is the message ID for the robust message you want to send?"
        mock_llm_json_output = {
            "intent": "Send robust message via Gemini",
            "mcp_tool_name": "RobustMCP",
            "parameters": {"method_name": "process_message"}, # Missing message_json
            "requires_further_clarification": True,
            "clarification_question": clarification_q,
            "response_to_user": clarification_q
        }
        self._mock_llm_generate_content_response(json.dumps(mock_llm_json_output))

        query = "I want to send a message robustly using Gemini."
        response = self.agent.process_query(query)

        self.assertEqual(response, clarification_q)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_for_query"), query)
        self.assertEqual(self.agent.get_factual_memory("last_clarification_question"), clarification_q)


if __name__ == '__main__':
    unittest.main()

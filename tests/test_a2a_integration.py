import unittest
import json
import uuid
import os
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow importing project modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from a2a_protocol import AgentCard, A2ATask, A2AClient
from openai_agent import OpenAIAgent # For testing agent specific A2A features
from anthropic_agent import AnthropicAgent # For testing agent specific A2A features

# Dummy MCP configs for agent initialization
AGENT_CONFIG = {
    "streaming_mcp_config": {},
    "federated_mcp_config": {},
    "robust_mcp_config": {}
}

# Mock API keys for testing agent instantiation without real keys
MOCK_OPENAI_API_KEY = "fake_openai_key"
MOCK_ANTHROPIC_API_KEY = "fake_anthropic_key"

class TestA2AProtocolComponents(unittest.TestCase):
    def test_agent_card_serialization(self):
        card = AgentCard(
            agent_id="test-agent-001",
            name="TestAgent",
            description="A test agent.",
            version="0.1.0",
            capabilities=[{"name": "test_skill", "description": "Performs a test."}],
            endpoint_url="http://localhost:1234/a2a"
        )
        card_dict = card.to_dict()
        self.assertEqual(card_dict["agentId"], "test-agent-001")
        self.assertEqual(card_dict["name"], "TestAgent")

        card_json = card.to_json()
        loaded_card_dict = json.loads(card_json)
        self.assertEqual(loaded_card_dict["agentId"], "test-agent-001")

        rehydrated_card = AgentCard.from_dict(card_dict)
        self.assertEqual(rehydrated_card.agent_id, "test-agent-001")
        self.assertEqual(rehydrated_card.name, "TestAgent")

    def test_a2a_task_serialization(self):
        task = A2ATask(
            task_id="test-task-001",
            client_agent_id="client-001",
            remote_agent_id="remote-001",
            capability_name="do_something",
            parameters={"param1": "value1"}
        )
        task_dict = task.to_dict()
        self.assertEqual(task_dict["taskId"], "test-task-001")
        self.assertEqual(task_dict["capabilityName"], "do_something")

        task_json = task.to_json()
        loaded_task_dict = json.loads(task_json)
        self.assertEqual(loaded_task_dict["taskId"], "test-task-001")

        rehydrated_task = A2ATask.from_dict(task_dict)
        self.assertEqual(rehydrated_task.task_id, "test-task-001")
        self.assertEqual(rehydrated_task.parameters["param1"], "value1")


@patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_OPENAI_API_KEY, "ANTHROPIC_API_KEY": MOCK_ANTHROPIC_API_KEY})
class TestAgentA2ACompliance(unittest.TestCase):

    def setUp(self):
        # This check is to prevent tests from running if keys are truly needed and not mocked away by agent logic
        if not MOCK_OPENAI_API_KEY:
            self.skipTest("Mock OpenAI API key is not set.")
        if not MOCK_ANTHROPIC_API_KEY:
            self.skipTest("Mock Anthropic API key is not set.")

        self.openai_agent = OpenAIAgent(api_key=MOCK_OPENAI_API_KEY, agent_config=AGENT_CONFIG)
        self.anthropic_agent = AnthropicAgent(api_key=MOCK_ANTHROPIC_API_KEY, agent_config=AGENT_CONFIG)


    def test_openai_agent_card_generation(self):
        card = self.openai_agent.get_agent_card()
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.agent_id, self.openai_agent.agent_id)
        self.assertTrue(len(card.capabilities) > 0) # Assuming it derives some from MCPs
        self.assertIn("OpenAI", card.name)
        self.assertEqual(card.endpoint_url, self.openai_agent.a2a_endpoint_url)

    def test_anthropic_agent_card_generation(self):
        card = self.anthropic_agent.get_agent_card()
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.agent_id, self.anthropic_agent.agent_id)
        self.assertTrue(len(card.capabilities) > 0)
        self.assertIn("Anthropic", card.name)
        self.assertEqual(card.endpoint_url, self.anthropic_agent.a2a_endpoint_url)

    def test_openai_handle_a2a_task_request_known_capability(self):
        agent_card = self.openai_agent.get_agent_card()
        if not agent_card.capabilities:
            self.skipTest("OpenAI agent has no capabilities listed for testing task handling.")

        capability_to_test = agent_card.capabilities[0]['name']
        task_data = {
            "jsonrpc": "2.0", "id": str(uuid.uuid4()),
            "method": "executeTask", # This would be the JSON-RPC method, not directly part of A2ATask usually
            "params": { # The A2ATask object itself is often the param
                "taskId": "sim-task-001",
                "clientAgentId": "test-client",
                "remoteAgentId": self.openai_agent.agent_id,
                "capabilityName": capability_to_test,
                "parameters": {"input": "test"}
            }
        }
        # The handle_a2a_task_request expects the A2ATask dict directly
        response = self.openai_agent.handle_a2a_task_request(task_data["params"])
        self.assertIn("result", response)
        self.assertEqual(response["result"]["status"], "completed")
        self.assertEqual(response["result"]["taskId"], "sim-task-001")

    def test_openai_handle_a2a_task_request_unknown_capability(self):
        task_data = {
            "taskId": "sim-task-002",
            "clientAgentId": "test-client",
            "remoteAgentId": self.openai_agent.agent_id,
            "capabilityName": "non_existent_capability_abc123",
            "parameters": {}
        }
        response = self.openai_agent.handle_a2a_task_request(task_data)
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601) # Method not found

# For the collaboration demo, we need to ensure it can run without actual API calls if possible,
# or mock the parts that require external services or environment variables not set in CI.
class TestCollaborationDemo(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_OPENAI_API_KEY, "ANTHROPIC_API_KEY": MOCK_ANTHROPIC_API_KEY})
    @patch('openai_agent.OpenAIAgent.process_query', return_value="Mocked OpenAI response") # Mock LLM calls
    @patch('anthropic_agent.AnthropicAgent.process_query', return_value="Mocked Anthropic response") # Mock LLM calls
    @patch('openai_agent.OpenAIAgent._determine_intent_and_params_with_llm') # Mock LLM calls deeper
    @patch('anthropic_agent.AnthropicAgent._determine_intent_and_params_with_llm') # Mock LLM calls deeper
    def test_collaboration_demo_runs(self, mock_anthropic_intent, mock_openai_intent,
                                   mock_anthropic_query, mock_openai_query):

        # Mock the intent calls to prevent actual LLM calls during agent init or card generation if any
        mock_openai_intent.return_value = {"intent": "mock", "mcp_tool_name": None, "parameters": {}}
        mock_anthropic_intent.return_value = {"intent": "mock", "mcp_tool_name": None, "parameters": {}}

        # Path to the demo script
        demo_script_path = os.path.join(os.path.dirname(__file__), '..', 'a2a_collaboration_demo.py')

        if not os.path.exists(demo_script_path):
            self.skipTest(f"Demo script not found at {demo_script_path}")

        # Mock stdout to capture logs/prints from the script
        from io import StringIO
        mock_stdout = StringIO()

        try:
            with patch('sys.stdout', new=mock_stdout):
                # Import the demo script as a module to run its main()
                import a2a_collaboration_demo
                # The demo's main is async
                import asyncio
                asyncio.run(a2a_collaboration_demo.main())

            output = mock_stdout.getvalue()
            self.assertIn("--- A2A Collaboration Demo Script ---", output)
            self.assertIn("OpenAI and Anthropic agents initialized.", output)
            self.assertIn("Simulating A2A call", output)
            self.assertIn("Response received by OpenAI Agent from Anthropic Agent", output)
            self.assertIn("--- End of A2A Collaboration Demo ---", output)
            self.assertNotIn("ERROR", output) # Check for major errors in output

        except Exception as e:
            self.fail(f"a2a_collaboration_demo.py failed to run: {e}\nOutput:\n{mock_stdout.getvalue()}")

if __name__ == '__main__':
    unittest.main()

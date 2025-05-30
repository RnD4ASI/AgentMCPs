import unittest
import json
import os

# Assuming mcp_configurations.json is in the parent directory of 'tests' (i.e., project root)
MCP_CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_configurations.json")

class TestMCPConfigurations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the MCP configuration once for all tests."""
        if not os.path.exists(MCP_CONFIG_FILE_PATH):
            raise FileNotFoundError(f"MCP configuration file not found at {MCP_CONFIG_FILE_PATH}")
        with open(MCP_CONFIG_FILE_PATH, 'r') as f:
            try:
                cls.mcp_config = json.load(f)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Failed to decode JSON from {MCP_CONFIG_FILE_PATH}: {e}")

    def test_top_level_structure(self):
        """Test presence of top-level keys."""
        self.assertIn("mcp_version", self.mcp_config)
        self.assertIsInstance(self.mcp_config["mcp_version"], str)

        self.assertIn("models", self.mcp_config)
        self.assertIsInstance(self.mcp_config["models"], list)

        self.assertIn("skills", self.mcp_config)
        self.assertIsInstance(self.mcp_config["skills"], list)

        self.assertIn("context_types", self.mcp_config)
        self.assertIsInstance(self.mcp_config["context_types"], list)

    def test_models_structure(self):
        """Test the structure of entries in the 'models' array."""
        self.assertTrue(len(self.mcp_config["models"]) >= 1, "Should be at least one model defined.")
        for model in self.mcp_config["models"]:
            self.assertIsInstance(model, dict)
            self.assertIn("id", model)
            self.assertIsInstance(model["id"], str)
            self.assertIn("name", model)
            self.assertIsInstance(model["name"], str)
            self.assertIn("type", model)
            self.assertIsInstance(model["type"], str)
            self.assertIn("provider", model)
            self.assertIsInstance(model["provider"], str)
            self.assertIn("api_details", model)
            self.assertIsInstance(model["api_details"], dict)
            self.assertIn("capabilities", model)
            self.assertIsInstance(model["capabilities"], list)
            # Check provider specific api_details
            if model["provider"] in ["AzureOpenAI", "OpenAI", "Anthropic", "GoogleGemini"]:
                self.assertIn("api_key_env_var", model["api_details"])
                self.assertIn("model_name", model["api_details"]) # or deployment_name_env_var for Azure
            elif model["provider"] == "HuggingFace":
                self.assertIn("model_name_or_path", model["api_details"])


    def test_skills_structure(self):
        """Test the structure of entries in the 'skills' array."""
        self.assertTrue(len(self.mcp_config["skills"]) >= 1, "Should be at least one skill defined.")
        for skill in self.mcp_config["skills"]:
            self.assertIsInstance(skill, dict)
            self.assertIn("id", skill)
            self.assertIsInstance(skill["id"], str)
            self.assertIn("name", skill)
            self.assertIsInstance(skill["name"], str)
            self.assertIn("description", skill)
            self.assertIsInstance(skill["description"], str)
            self.assertIn("handler_module", skill)
            self.assertIsInstance(skill["handler_module"], str)
            self.assertIn("handler_class_or_function", skill)
            self.assertIsInstance(skill["handler_class_or_function"], str)
            self.assertIn("parameters", skill)
            self.assertIsInstance(skill["parameters"], list)
            for param in skill["parameters"]:
                self.assertIn("name", param)
                self.assertIn("type", param)
                self.assertIn("description", param)
                self.assertIn("is_required", param)
                self.assertIsInstance(param["is_required"], bool)
            self.assertIn("expected_context", skill) # Can be empty list
            self.assertIsInstance(skill["expected_context"], list)
            self.assertIn("output_schema", skill)
            self.assertIsInstance(skill["output_schema"], dict)
            self.assertIn("type", skill["output_schema"]) # Basic JSON schema check

    def test_context_types_structure(self):
        """Test the structure of entries in the 'context_types' array."""
        self.assertTrue(len(self.mcp_config["context_types"]) >= 1, "Should be at least one context_type defined.")
        for context_type in self.mcp_config["context_types"]:
            self.assertIsInstance(context_type, dict)
            self.assertIn("id", context_type)
            self.assertIsInstance(context_type["id"], str)
            self.assertIn("description", context_type)
            self.assertIsInstance(context_type["description"], str)
            self.assertIn("schema", context_type)
            self.assertIsInstance(context_type["schema"], dict)
            self.assertIn("type", context_type["schema"]) # Basic JSON schema check

    def test_specific_skill_definitions_exist(self):
        """Test that the three specific skills are defined."""
        skill_ids = [skill["id"] for skill in self.mcp_config["skills"]]
        self.assertIn("realtime_data_processor", skill_ids)
        self.assertIn("federated_model_aggregator", skill_ids)
        self.assertIn("robust_message_handler", skill_ids)

    def test_specific_context_type_definitions_exist(self):
        """Test that specific context types for memory are defined."""
        context_ids = [ct["id"] for ct in self.mcp_config["context_types"]]
        self.assertIn("factual_memory_snapshot", context_ids)
        self.assertIn("procedural_memory_snapshot", context_ids)
        self.assertIn("user_query_context", context_ids) # Example other context

if __name__ == '__main__':
    unittest.main()

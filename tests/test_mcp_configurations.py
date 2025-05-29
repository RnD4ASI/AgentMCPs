import unittest
import json
import os

# Assuming mcp_configurations.json is in the parent directory of 'tests' (i.e., project root)
MCP_CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_configurations.json")

class TestMCPConfigurations(unittest.TestCase):

    def test_load_mcp_configurations_file(self):
        """Test if the MCP configurations file loads and is valid JSON."""
        self.assertTrue(os.path.exists(MCP_CONFIG_FILE_PATH), f"MCP configuration file not found at {MCP_CONFIG_FILE_PATH}")
        
        with open(MCP_CONFIG_FILE_PATH, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                self.fail(f"Failed to decode JSON from {MCP_CONFIG_FILE_PATH}: {e}")
        
        self.assertIsInstance(data, list, "MCP configurations should be a list.")
        self.assertTrue(len(data) >= 1, "MCP configurations should have at least one entry (actually 5 as per problem spec, but testing for >=1).")

    def test_mcp_configuration_structure(self):
        """Test the basic structure of each MCP configuration entry."""
        self.assertTrue(os.path.exists(MCP_CONFIG_FILE_PATH), "MCP configuration file not found.")
        
        with open(MCP_CONFIG_FILE_PATH, 'r') as f:
            configs = json.load(f)

        for config in configs:
            self.assertIsInstance(config, dict, "Each configuration entry should be a dictionary.")
            self.assertIn("name", config, "Each configuration must have a 'name' field.")
            self.assertIsInstance(config["name"], str, "'name' field should be a string.")
            
            self.assertIn("description", config, "Each configuration must have a 'description' field.")
            self.assertIsInstance(config["description"], str, "'description' field should be a string.")

            self.assertIn("protocol_details", config, "Each configuration must have 'protocol_details'.")
            self.assertIsInstance(config["protocol_details"], dict, "'protocol_details' should be a dictionary.")
            
            self.assertIn("type", config["protocol_details"], "'protocol_details' must have a 'type' field.")
            self.assertIsInstance(config["protocol_details"]["type"], str, "'type' in 'protocol_details' should be a string.")

            # Optional but common fields to check type if they exist
            if "security_config" in config:
                self.assertIsInstance(config["security_config"], dict, "'security_config' should be a dictionary if present.")
            if "data_format_preferences" in config:
                self.assertIsInstance(config["data_format_preferences"], dict, "'data_format_preferences' should be a dictionary if present.")

if __name__ == '__main__':
    unittest.main()

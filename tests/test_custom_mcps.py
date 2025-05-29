import unittest
from unittest.mock import patch, MagicMock, call
import json
import sys
import os

# Add project root to sys.path to allow importing custom_mcp modules
# Assumes tests are run from the project root or that the 'tests' directory is directly under the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_mcp_1 import RealTimeStreamingMCP
from custom_mcp_2 import FederatedLearningMCP
from custom_mcp_3 import RobustMCP

class TestCustomMCPs(unittest.TestCase):

    # --- Tests for RealTimeStreamingMCP (custom_mcp_1.py) ---
    def test_streaming_mcp_initialization(self):
        config = {"validation_rules": {}, "transformation_map": {}}
        mcp = RealTimeStreamingMCP(config=config)
        self.assertIsNotNone(mcp)
        self.assertEqual(mcp.config, config)

    def test_streaming_mcp_validation(self):
        config = {
            'validation_rules': {
                'temperature': {'type': float, 'min': -10.0, 'max': 40.0},
                'humidity': {'type': int, 'min': 0, 'max': 100}
            }
        }
        mcp = RealTimeStreamingMCP(config=config)
        
        valid_data = {"temperature": 25.5, "humidity": 60}
        is_valid, errors = mcp._validate_data(valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        invalid_data_temp_type = {"temperature": "hot", "humidity": 60}
        is_valid, errors = mcp._validate_data(invalid_data_temp_type)
        self.assertFalse(is_valid)
        self.assertIn("Invalid type for temperature", errors[0])

        invalid_data_humidity_range = {"temperature": 20.0, "humidity": 101}
        is_valid, errors = mcp._validate_data(invalid_data_humidity_range)
        self.assertFalse(is_valid)
        self.assertIn("greater than max", errors[0])
        
        missing_field_data = {"temperature": 20.0}
        is_valid, errors = mcp._validate_data(missing_field_data)
        self.assertFalse(is_valid)
        self.assertIn("Missing field: humidity", errors[0])


    def test_streaming_mcp_transformation(self):
        config = {
            'transformation_map': {
                'temp_c': lambda c: (c * 9/5) + 32, # transform to fahrenheit and rename
                'add_timestamp': True
            }
        }
        mcp = RealTimeStreamingMCP(config=config)
        # Note: The lambda in config is for 'temp_c', but the original class example used 'temperature_celsius'.
        # The test needs to align with how the _transform_data method actually looks up keys.
        # The provided custom_mcp_1.py has specific logic for 'temperature_celsius'.
        # Let's adjust the test to match the provided MCP's behavior or assume a more generic transform_map.
        # For this test, I'll assume the transformation_map key matches the input data key.
        
        # Re-checking custom_mcp_1.py, it specifically checks for 'temperature_celsius'.
        # So the transformation_map key should be 'temperature_celsius' for that logic to trigger.
        config_for_actual_mcp = {
             'transformation_map': {
                'temperature_celsius': lambda c: (c * 9/5) + 32, 
                'add_timestamp': True
            }
        }
        mcp_actual = RealTimeStreamingMCP(config=config_for_actual_mcp)
        data_item = {"sensor": "S1", "temperature_celsius": 20.0}
        transformed = mcp_actual._transform_data(data_item)
        
        self.assertIn("temperature_fahrenheit", transformed) # It renames to fahrenheit
        self.assertAlmostEqual(transformed["temperature_fahrenheit"], (20.0 * 9/5) + 32)
        self.assertNotIn("temperature_celsius", transformed) # Original key is removed
        self.assertIn("processing_timestamp_utc", transformed)


    def test_streaming_mcp_process_stream_data(self):
        config = {
            'validation_rules': {'value': {'type': int, 'min': 0}},
            'transformation_map': {'add_timestamp': True}
        }
        mcp = RealTimeStreamingMCP(config=config)
        mock_callback = MagicMock()
        mcp.register_data_callback(mock_callback)

        raw_json_valid = '[{"id": "item1", "value": 10}, {"id": "item2", "value": 20}]'
        mcp.process_stream_data(raw_json_valid)
        self.assertEqual(mock_callback.call_count, 2)
        # Check first call's argument (simplified check)
        args_item1, _ = mock_callback.call_args_list[0]
        self.assertEqual(args_item1[0]['id'], 'item1')
        self.assertIn('processing_timestamp_utc', args_item1[0])

        mock_callback.reset_mock()
        raw_json_invalid = '[{"id": "item3", "value": -5}]' # Fails validation
        mcp.process_stream_data(raw_json_invalid)
        mock_callback.assert_not_called()
        
        raw_json_mixed = '[{"id": "item4", "value": 30}, {"id": "item5", "value": -1}]'
        mcp.process_stream_data(raw_json_mixed)
        mock_callback.assert_called_once() # Only item4 should pass


    # --- Tests for FederatedLearningMCP (custom_mcp_2.py) ---
    def test_federated_mcp_initialization(self):
        mcp = FederatedLearningMCP(global_model_id="test_model_fl")
        self.assertEqual(mcp.global_model_id, "test_model_fl")
        self.assertIn("weights", mcp.global_model_parameters)
        self.assertEqual(mcp.current_round, 0)

    def test_federated_mcp_start_new_round(self):
        mcp = FederatedLearningMCP(global_model_id="test_model_fl")
        initial_params = mcp.global_model_parameters.copy()
        
        mcp.start_new_round(round_id=1)
        self.assertEqual(mcp.current_round, 1)
        self.assertIn(1, mcp.participant_updates)
        self.assertEqual(mcp.global_model_parameters, initial_params) # Params shouldn't change on start_new_round

    def test_federated_mcp_submit_model_update(self):
        mcp = FederatedLearningMCP(global_model_id="test_model_fl")
        mcp.start_new_round(round_id=1)
        
        update_data = {"weights_delta": [0.1, 0.2], "bias_delta": 0.05, "data_samples_count": 100}
        mcp.submit_model_update("p1", 1, json.dumps(update_data))
        self.assertIn("p1", mcp.participant_updates[1])
        self.assertEqual(mcp.participant_updates[1]["p1"]["data_samples_count"], 100)

        # Test submitting for wrong round
        mcp.submit_model_update("p2", 0, json.dumps(update_data)) # current_round is 1
        self.assertNotIn("p2", mcp.participant_updates[1])


    def test_federated_mcp_aggregation_average(self):
        mcp = FederatedLearningMCP(global_model_id="test_model_avg", aggregation_strategy='average')
        mcp.global_model_parameters = {"weights": [1.0, 2.0], "bias": 0.5} # Known initial state
        mcp.start_new_round(round_id=1)

        update1 = {"weights_delta": [0.2, 0.4], "bias_delta": 0.1, "data_samples_count": 100}
        update2 = {"weights_delta": [0.4, 0.6], "bias_delta": -0.1, "data_samples_count": 100}
        mcp.submit_model_update("p1", 1, json.dumps(update1))
        mcp.submit_model_update("p2", 1, json.dumps(update2))

        mcp.aggregate_updates(round_id=1)
        # Expected: weights = [1 + (0.2+0.4)/2, 2 + (0.4+0.6)/2] = [1.3, 2.5]
        # Expected: bias = 0.5 + (0.1-0.1)/2 = 0.5
        self.assertAlmostEqual(mcp.global_model_parameters["weights"][0], 1.3)
        self.assertAlmostEqual(mcp.global_model_parameters["weights"][1], 2.5)
        self.assertAlmostEqual(mcp.global_model_parameters["bias"], 0.5)

    def test_federated_mcp_aggregation_weighted_average(self):
        mcp = FederatedLearningMCP(global_model_id="test_model_weighted", aggregation_strategy='weighted_average')
        mcp.global_model_parameters = {"weights": [1.0], "bias": 0.0} # Simpler model
        mcp.start_new_round(round_id=1)

        update1 = {"weights_delta": [0.1], "bias_delta": 0.02, "data_samples_count": 100} # weight 100
        update2 = {"weights_delta": [0.3], "bias_delta": 0.06, "data_samples_count": 300} # weight 300
        mcp.submit_model_update("p1", 1, json.dumps(update1))
        mcp.submit_model_update("p2", 1, json.dumps(update2))

        mcp.aggregate_updates(round_id=1)
        # Total samples = 400
        # Expected weights_delta_agg = (0.1*100 + 0.3*300) / 400 = (10 + 90) / 400 = 100 / 400 = 0.25
        # Expected bias_delta_agg = (0.02*100 + 0.06*300) / 400 = (2 + 18) / 400 = 20 / 400 = 0.05
        # Expected new weights = [1.0 + 0.25] = [1.25]
        # Expected new bias = 0.0 + 0.05 = 0.05
        self.assertAlmostEqual(mcp.global_model_parameters["weights"][0], 1.25)
        self.assertAlmostEqual(mcp.global_model_parameters["bias"], 0.05)


    # --- Tests for RobustMCP (custom_mcp_3.py) ---
    @patch('custom_mcp_3.time.sleep', return_value=None) # Mock time.sleep
    def test_robust_mcp_initialization_and_success(self, mock_sleep):
        mock_processor = MagicMock(return_value={'status': 'success', 'result': 'Processed!'})
        mock_dlq = MagicMock()
        config = {
            'max_retries': 2, 
            'base_retry_delay_seconds': 0.1,
            'processing_function': mock_processor,
            'dlq_handler': mock_dlq
        }
        mcp = RobustMCP(config=config)
        
        message_payload = {"id": "msg1", "payload": {"data": "test_data"}}
        mcp.process_message(json.dumps(message_payload))

        mock_processor.assert_called_once_with({"data": "test_data"})
        mock_dlq.assert_not_called()
        mock_sleep.assert_not_called() # No retries on success

    @patch('custom_mcp_3.time.sleep', return_value=None)
    def test_robust_mcp_transient_failure_then_success(self, mock_sleep):
        mock_processor = MagicMock(side_effect=[
            {'status': 'failure', 'error_type': 'transient', 'message': 'Attempt 1 fail'},
            {'status': 'success', 'result': 'Processed on attempt 2'}
        ])
        mock_dlq = MagicMock()
        config = {'max_retries': 2, 'processing_function': mock_processor, 'dlq_handler': mock_dlq}
        mcp = RobustMCP(config=config)
        
        message_payload = {"id": "msg_transient", "payload": {"data": "flaky_data"}}
        mcp.process_message(json.dumps(message_payload))

        self.assertEqual(mock_processor.call_count, 2)
        mock_dlq.assert_not_called()
        mock_sleep.assert_called_once() # Should have slept after the first failure

    @patch('custom_mcp_3.time.sleep', return_value=None)
    def test_robust_mcp_permanent_failure_to_dlq(self, mock_sleep):
        mock_processor = MagicMock(return_value={'status': 'failure', 'error_type': 'permanent', 'message': 'Bad data'})
        mock_dlq = MagicMock()
        config = {'max_retries': 2, 'processing_function': mock_processor, 'dlq_handler': mock_dlq}
        mcp = RobustMCP(config=config)
        
        message_payload = {"id": "msg_permanent", "payload": {"data": "corrupt_data"}}
        mcp.process_message(json.dumps(message_payload))

        mock_processor.assert_called_once()
        mock_dlq.assert_called_once_with(message_payload, "Permanent failure: Bad data")
        mock_sleep.assert_not_called()

    @patch('custom_mcp_3.time.sleep', return_value=None)
    def test_robust_mcp_max_retries_to_dlq(self, mock_sleep):
        mock_processor = MagicMock(side_effect=[
            {'status': 'failure', 'error_type': 'transient', 'message': 'Fail 1'},
            {'status': 'failure', 'error_type': 'transient', 'message': 'Fail 2'},
            {'status': 'failure', 'error_type': 'transient', 'message': 'Fail 3'} 
        ]) # Max retries is 2, so 3 calls total (initial + 2 retries)
        mock_dlq = MagicMock()
        config = {
            'max_retries': 2, 
            'base_retry_delay_seconds': 0.01, # small delay for test speed
            'processing_function': mock_processor, 
            'dlq_handler': mock_dlq
        }
        mcp = RobustMCP(config=config)
        
        message_payload = {"id": "msg_max_retry", "payload": {"data": "very_flaky_data"}}
        mcp.process_message(json.dumps(message_payload))

        self.assertEqual(mock_processor.call_count, 3) # Initial attempt + 2 retries
        mock_dlq.assert_called_once_with(message_payload, "Max retries reached after transient failures. Last error: Fail 3")
        self.assertEqual(mock_sleep.call_count, 2) # Slept after first and second failures
        # Check exponential backoff (simplified, just checking call count)
        # mock_sleep.assert_any_call(0.01 * (2**0) + ANY) # Basic check for delay logic if needed


if __name__ == '__main__':
    unittest.main()

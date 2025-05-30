import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os
import time # For robust_messaging_skill direct time.sleep patch if needed for its own tests

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the refactored skill classes/handlers
# The handler_class_or_function in mcp_configurations.json points to these specific names.
from realtime_processing_skill import RealTimeStreamingMCP as RealtimeDataProcessorSkillHandler # Alias for clarity if preferred, or use original
from federated_learning_skill import FederatedLearningMCP as FederatedLearningSkillHandler
from robust_messaging_skill import RobustMCP as RobustMessagingSkillHandler


class TestMCPSkills(unittest.TestCase):

    # --- Tests for RealtimeDataProcessorSkill (realtime_processing_skill.py) ---
    def test_realtime_skill_initialization(self):
        handler = RealtimeDataProcessorSkillHandler()
        self.assertIsNotNone(handler)

    def test_realtime_skill_process_stream_success(self):
        handler = RealtimeDataProcessorSkillHandler()
        params = {
            "raw_data_json": json.dumps([
                {"id": "s1", "temperature_celsius": 25.0},
                {"id": "s2", "temperature_celsius": 10.0}
            ])
        }
        context = { # Context providing validation and transformation rules
            "streaming_data_input_schema": {
                "schema": {
                    "properties": {
                        "id": {"type": "string", "is_required": True},
                        "temperature_celsius": {"type": "number"}
                    }
                }
            },
            "transformation_rules": {
                "celsius_to_fahrenheit": {"apply": True},
                "add_processing_timestamp": True
            }
        }
        response = handler.process_stream_data(parameters=params, context=context)

        self.assertEqual(response["status"], "success")
        self.assertIn("data", response)
        self.assertEqual(len(response["data"]["processed_items"]), 2)
        self.assertIn("temperature_fahrenheit", response["data"]["processed_items"][0])
        self.assertAlmostEqual(response["data"]["processed_items"][0]["temperature_fahrenheit"], (25.0 * 9/5) + 32)
        self.assertIn("skill_processing_timestamp_utc", response["data"]["processed_items"][0])
        self.assertEqual(response["data"]["summary"], "Processed 2/2 items.")

    def test_realtime_skill_validation_failure(self):
        handler = RealtimeDataProcessorSkillHandler()
        params = {
            "raw_data_json": json.dumps([{"id": "s1", "temperature_celsius": "hot"}]) # Invalid type
        }
        context = {
            "streaming_data_input_schema": {
                "schema": {
                    "properties": {"temperature_celsius": {"type": "number"}}
                }
            }
        }
        response = handler.process_stream_data(parameters=params, context=context)
        self.assertEqual(response["status"], "error") # Should be error as all items fail
        self.assertIn("errors_during_processing", response["data"])
        self.assertTrue(any("failed validation" in err for err in response["data"]["errors_during_processing"]))
        self.assertEqual(response["data"]["processed_items"], [])

    def test_realtime_skill_missing_parameter(self):
        handler = RealtimeDataProcessorSkillHandler()
        response = handler.process_stream_data(parameters={}, context={}) # Missing raw_data_json
        self.assertEqual(response["status"], "error")
        self.assertIn("Missing 'raw_data_json'", response["error_message"])


    # --- Tests for FederatedLearningSkill (federated_learning_skill.py) ---
    def test_federated_skill_initialization(self):
        handler = FederatedLearningSkillHandler()
        self.assertIsNotNone(handler)

    def test_federated_skill_start_round(self):
        handler = FederatedLearningSkillHandler()
        params = {"action": "start_new_round", "round_id": 1, "global_model_id": "test_fl_model"}
        context = {} # Initial context
        response = handler.execute_action(parameters=params, context=context)

        self.assertEqual(response["status"], "success")
        self.assertIn("data", response)
        self.assertIn("Round 1 started", response["data"]["message"])
        self.assertIn("global_model_parameters", response["data"])
        self.assertEqual(response["data"]["current_round_id"], 1)
        self.assertIn("context_updates_suggestion", response)
        self.assertEqual(response["context_updates_suggestion"]["current_round_id"], 1)

    def test_federated_skill_submit_update(self):
        handler = FederatedLearningSkillHandler()
        # Context simulating a round already started
        current_context = {
            "current_round_id": 1,
            "global_model_parameters": {"weights": [0.1], "bias": 0.01}, # Dummy model
            "participant_updates_round_1": {}
        }
        update_data = {"weights_delta": [0.05], "bias_delta": 0.005, "data_samples_count": 50}
        params = {
            "action": "submit_model_update",
            "round_id": 1,
            "participant_id": "p1",
            "model_update_json": json.dumps(update_data)
        }
        response = handler.execute_action(parameters=params, context=current_context)

        self.assertEqual(response["status"], "success")
        self.assertIn("Participant p1's update for round 1 received", response["data"]["message"])
        self.assertIn("context_updates_suggestion", response)
        self.assertIn("p1", response["context_updates_suggestion"]["participant_updates_round_1"])


    def test_federated_skill_aggregate_updates_weighted_avg(self):
        handler = FederatedLearningSkillHandler()
        current_context = {
            "current_round_id": 1,
            "global_model_parameters": {"weights": [1.0], "bias": 0.1},
            "participant_updates_round_1": { # Data already submitted for this round
                "pA": {"weights_delta": [0.2], "bias_delta": 0.02, "data_samples_count": 100},
                "pB": {"weights_delta": [0.4], "bias_delta": 0.04, "data_samples_count": 300}
            }
        }
        params = {"action": "aggregate_updates", "round_id": 1, "aggregation_strategy": "weighted_average"}
        response = handler.execute_action(parameters=params, context=current_context)

        self.assertEqual(response["status"], "success")
        self.assertIn("Global model updated", response["data"]["message"])
        # Expected new weight: 1.0 + ((0.2*100 + 0.4*300) / 400) = 1.0 + (20+120)/400 = 1.0 + 140/400 = 1.0 + 0.35 = 1.35
        self.assertAlmostEqual(response["data"]["global_model_parameters"]["weights"][0], 1.35)
        # Expected new bias: 0.1 + ((0.02*100 + 0.04*300) / 400) = 0.1 + (2+12)/400 = 0.1 + 14/400 = 0.1 + 0.035 = 0.135
        self.assertAlmostEqual(response["data"]["global_model_parameters"]["bias"], 0.135)
        self.assertIn("context_updates_suggestion", response)
        self.assertEqual(response["context_updates_suggestion"]["global_model_parameters"]["weights"][0], 1.35)


    # --- Tests for RobustMessagingSkill (robust_messaging_skill.py) ---
    def test_robust_skill_initialization(self):
        handler = RobustMessagingSkillHandler()
        self.assertIsNotNone(handler)

    @patch('robust_messaging_skill.time.sleep', return_value=None) # Mock time.sleep for retries
    def test_robust_skill_success(self, mock_sleep):
        # Mock the processing function within the context for predictable outcome
        mock_processor = MagicMock(return_value={'status': 'success', 'result': 'Test processed!'})
        context = {"custom_processing_function": mock_processor, "dlq_target_info": "TestDLQ"}

        handler = RobustMessagingSkillHandler(skill_config={'default_max_retries': 1})
        params = {"message_json": json.dumps({"id": "msg1", "payload": {"data": "content"}})}

        response = handler.process_message(parameters=params, context=context)

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["data"]["message_id"], "msg1")
        self.assertEqual(response["data"]["result"], "Test processed!")
        mock_processor.assert_called_once_with({"data": "content"}, context) # Ensure context is passed
        mock_sleep.assert_not_called()

    @patch('robust_messaging_skill.time.sleep', return_value=None)
    def test_robust_skill_permanent_failure_to_dlq(self, mock_sleep):
        mock_processor = MagicMock(return_value={'status': 'failure', 'error_type': 'permanent', 'message': 'Fatal error'})
        mock_dlq_handler = MagicMock() # Mock DLQ handler passed in context
        context = {
            "custom_processing_function": mock_processor,
            "custom_dlq_handler": mock_dlq_handler,
            "dlq_target_info": "CustomTestDLQ"
        }

        handler = RobustMessagingSkillHandler()
        message_content = {"id": "msg_perm", "payload": {"critical": True}}
        params = {"message_json": json.dumps(message_content)}

        response = handler.process_message(parameters=params, context=context)

        self.assertEqual(response["status"], "error")
        self.assertIn("Permanent failure", response["error_message"])
        self.assertEqual(response["data"]["action_taken"], "sent_to_dlq_permanent_failure")
        mock_processor.assert_called_once()
        # Check that our custom DLQ handler was called
        mock_dlq_handler.assert_called_once_with(message_content, "Permanent failure after 1 attempt(s): Fatal error", context)
        mock_sleep.assert_not_called()


    @patch('robust_messaging_skill.time.sleep', return_value=None)
    def test_robust_skill_max_retries_to_dlq(self, mock_sleep):
        # Simulate transient failures for all attempts
        mock_processor = MagicMock(side_effect=[
            {'status': 'failure', 'error_type': 'transient', 'message': 'Attempt 1 failed'},
            {'status': 'failure', 'error_type': 'transient', 'message': 'Attempt 2 failed'},
            {'status': 'failure', 'error_type': 'transient', 'message': 'Attempt 3 failed'}
        ])
        mock_dlq_handler = MagicMock()
        context = {"custom_processing_function": mock_processor, "custom_dlq_handler": mock_dlq_handler}

        # Max retries = 2 (so 3 total attempts: initial + 2 retries)
        handler = RobustMessagingSkillHandler(skill_config={'default_max_retries': 2, 'default_base_retry_delay_seconds': 0.01})
        message_content = {"id": "msg_retry_max", "payload": {"info": "flaky"}}
        params = {"message_json": json.dumps(message_content)}
        # Can also pass 'max_retries' in params to override skill_config for this call

        response = handler.process_message(parameters=params, context=context)

        self.assertEqual(response["status"], "error")
        self.assertIn("Max retries (3) reached", response["error_message"])
        self.assertEqual(response["data"]["action_taken"], "sent_to_dlq_max_retries")
        self.assertEqual(mock_processor.call_count, 3) # Initial + 2 retries
        self.assertEqual(mock_sleep.call_count, 2) # Slept after 1st and 2nd failures
        mock_dlq_handler.assert_called_once_with(message_content, "Max retries (3) reached for message ID 'msg_retry_max'. Last error: Attempt 3 failed", context)


if __name__ == '__main__':
    unittest.main()

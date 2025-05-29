import time
import json
import random

class RobustMCP:
    """
    MCP designed for robust message processing with retry mechanisms
    for transient errors and dead-letter queue (DLQ) integration for
    persistent failures.
    """

    def __init__(self, config):
        """
        Initializes the RobustMCP.

        Args:
            config (dict): Configuration for the MCP.
                           Expected keys:
                           - 'max_retries': Max number of retries for a message. (int)
                           - 'base_retry_delay_seconds': Initial delay for retries. (float)
                           - 'dlq_handler': A function to call for messages sent to DLQ. (function)
                           - 'processing_function': A function that processes the message.
                                                    This function must take message_content (dict)
                                                    as input and return a dict with 'status': 'success'
                                                    or 'status': 'failure', 'error_type': 'transient'/'permanent',
                                                    'message': 'error description'.
        """
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.base_retry_delay_seconds = config.get('base_retry_delay_seconds', 1.0)
        self.dlq_handler = config.get('dlq_handler', self._default_dlq_handler)
        self.processing_function = config.get('processing_function', self._default_processing_function)
        
        print(f"RobustMCP initialized. Max retries: {self.max_retries}, Base delay: {self.base_retry_delay_seconds}s")

    def _default_dlq_handler(self, message, reason):
        """Default handler for messages that are sent to the DLQ."""
        print(f"DLQ: Message ID '{message.get('id', 'N/A')}' sent to DLQ. Reason: {reason}")
        # In a real system, this would integrate with an actual DLQ service (e.g., SQS DLQ, Kafka dead letter topic)

    def _default_processing_function(self, message_content):
        """
        Default (simulated) processing function.
        This function simulates successes, transient errors, and permanent errors.
        """
        # Simulate different outcomes
        rand_val = random.random()
        if 'force_failure_type' in message_content: # For testing
            if message_content['force_failure_type'] == 'transient':
                rand_val = 0.4
            elif message_content['force_failure_type'] == 'permanent':
                rand_val = 0.1
        
        if rand_val < 0.2: # 20% chance of permanent failure
            return {'status': 'failure', 'error_type': 'permanent', 'message': 'Simulated permanent error (e.g., invalid data)'}
        elif rand_val < 0.6: # 40% chance of transient failure (20% to 60%)
            return {'status': 'failure', 'error_type': 'transient', 'message': 'Simulated transient error (e.g., temporary network issue)'}
        else: # 40% chance of success
            return {'status': 'success', 'result': f"Successfully processed data: {message_content.get('data')}"}

    def process_message(self, message_json):
        """
        Processes a message with retry logic and DLQ integration.

        Args:
            message_json (str): JSON string representing the message.
                                Expected to have at least an 'id' and 'payload'.
                                Example: '{"id": "msg123", "payload": {"data": "some_value"}}'
        """
        try:
            message = json.loads(message_json)
            message_id = message.get('id', f"unknown_{time.time()}")
            payload = message.get('payload', {})
        except json.JSONDecodeError as e:
            print(f"Error decoding message JSON: {e}. Message: {message_json}")
            self.dlq_handler({"raw_message": message_json}, f"JSONDecodeError: {e}")
            return

        print(f"\nProcessing message ID: {message_id}, Payload: {payload}")
        
        for attempt in range(self.max_retries + 1):
            print(f"Attempt {attempt + 1}/{self.max_retries + 1} for message ID: {message_id}")
            try:
                result = self.processing_function(payload)

                if result.get('status') == 'success':
                    print(f"SUCCESS: Message ID '{message_id}' processed successfully. Result: {result.get('result')}")
                    return # Successfully processed
                
                error_type = result.get('error_type', 'unknown')
                error_message = result.get('message', 'No error message provided.')

                if error_type == 'permanent':
                    print(f"FAILURE (Permanent): Message ID '{message_id}'. Error: {error_message}. Sending to DLQ.")
                    self.dlq_handler(message, f"Permanent failure: {error_message}")
                    return # Permanent failure, send to DLQ

                # Transient failure, will retry if attempts left
                print(f"FAILURE (Transient): Message ID '{message_id}'. Error: {error_message}.")
                if attempt < self.max_retries:
                    delay = self.base_retry_delay_seconds * (2 ** attempt) # Exponential backoff
                    jitter = random.uniform(0, delay * 0.1) # Add jitter
                    actual_delay = delay + jitter
                    print(f"Retrying in {actual_delay:.2f} seconds...")
                    time.sleep(actual_delay)
                else:
                    print(f"Max retries reached for message ID '{message_id}'. Sending to DLQ.")
                    self.dlq_handler(message, f"Max retries reached after transient failures. Last error: {error_message}")
                    return

            except Exception as e: # Catch unexpected errors in processing_function
                print(f"CRITICAL FAILURE: Unexpected error during processing of message ID '{message_id}': {e}")
                if attempt < self.max_retries:
                    delay = self.base_retry_delay_seconds * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.1)
                    actual_delay = delay + jitter
                    print(f"Retrying in {actual_delay:.2f} seconds due to critical error...")
                    time.sleep(actual_delay)
                else:
                    print(f"Max retries reached for message ID '{message_id}' after critical error. Sending to DLQ.")
                    self.dlq_handler(message, f"Critical unexpected error: {e}")
                    return
        # Should not be reached if logic is correct
        print(f"Warning: Message ID '{message_id}' exited processing loop unexpectedly.")


def custom_processor_example(message_content):
    """An example of a custom processing function that can be passed to RobustMCP."""
    data_value = message_content.get("data_value", 0)
    if data_value < 0:
        return {'status': 'failure', 'error_type': 'permanent', 'message': f'Invalid data_value: {data_value} cannot be negative.'}
    if data_value % 10 == 0 and data_value > 0: # Simulate transient error for values ending in 0
        # Simulate a fix after a few tries
        if message_content.get("attempt_count", 0) < 2: # Fails first 2 times
             message_content["attempt_count"] = message_content.get("attempt_count", 0) + 1
             return {'status': 'failure', 'error_type': 'transient', 'message': f'Simulated network glitch for data_value {data_value}. Attempt: {message_content["attempt_count"]}'}
    return {'status': 'success', 'result': f'Custom processed data_value: {data_value}'}


if __name__ == '__main__':
    print("--- RobustMCP Demo with Default Processor ---")
    
    # Basic config using default processor and DLQ handler
    default_mcp_config = {
        'max_retries': 2,
        'base_retry_delay_seconds': 0.5,
    }
    robust_mcp_default = RobustMCP(config=default_mcp_config)

    # Message that should succeed eventually (or by chance with default processor)
    msg1_payload = {"data": "critical_info", "id": "msg001"}
    robust_mcp_default.process_message(json.dumps(msg1_payload))

    # Message forced to have a permanent error
    msg2_payload = {"id": "msg002", "payload": {"data": "bad_data", "force_failure_type": "permanent"}}
    robust_mcp_default.process_message(json.dumps(msg2_payload))

    # Message forced to have transient errors until DLQ
    msg3_payload = {"id": "msg003", "payload": {"data": "flaky_service_data", "force_failure_type": "transient"}}
    robust_mcp_default.process_message(json.dumps(msg3_payload))
    
    print("\n--- RobustMCP Demo with Custom Processor ---")
    
    def my_dlq(message, reason):
        print(f"MY_CUSTOM_DLQ: Message {message.get('id')} failed. Reason: {reason}")

    custom_mcp_config = {
        'max_retries': 3,
        'base_retry_delay_seconds': 0.2,
        'dlq_handler': my_dlq,
        'processing_function': custom_processor_example
    }
    robust_mcp_custom = RobustMCP(config=custom_mcp_config)

    # Message that will be processed by custom function - should succeed
    msg_custom_1 = {"id": "custom001", "payload": {"data_value": 5}}
    robust_mcp_custom.process_message(json.dumps(msg_custom_1))

    # Message that will hit permanent error in custom function
    msg_custom_2 = {"id": "custom002", "payload": {"data_value": -10}}
    robust_mcp_custom.process_message(json.dumps(msg_custom_2))

    # Message that will hit transient error in custom function and retry (and succeed)
    msg_custom_3 = {"id": "custom003", "payload": {"data_value": 20}} # Will fail twice, then succeed
    robust_mcp_custom.process_message(json.dumps(msg_custom_3))
    
    # Message that will hit transient error and exhaust retries
    # To ensure this, we need to make sure 'attempt_count' doesn't get reset,
    # The custom_processor_example as written will succeed on the 3rd try if max_retries >=2.
    # Let's simulate it by making it always fail transiently in the custom processor if we want to test DLQ for transient.
    # For this demo, msg_custom_3 above already shows retry. To show DLQ for transient with custom,
    # the custom_processor would need to be more stateful or designed to always fail transiently for a specific input.
    # The default processor's msg3_payload already demonstrates DLQ for transient failures.

    print("\n--- End of RobustMCP Demo ---")

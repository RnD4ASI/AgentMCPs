import time
import json
import random
import logging

# Setup basic logging for the skill
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RobustMCP: # Matching class name from mcp_configurations.json
    """
    MCP Skill for robust message processing with retry mechanisms
    and dead-letter queue (DLQ) integration.
    Matches 'robust_message_handler' skill in mcp_configurations.json.
    Handler method: 'process_message'
    """

    def __init__(self, skill_config=None):
        """
        Initializes the RobustMessageHandlerSkill.

        Args:
            skill_config (dict, optional): Configuration for the skill.
                                           Can include default_max_retries, default_base_retry_delay,
                                           default_processing_function, default_dlq_handler.
        """
        self.skill_config = skill_config if skill_config else {}
        logger.info("RobustMCP skill initialized.")

    def _default_simulated_dlq_handler(self, message_content: dict, reason: str, context: dict):
        """
        Default simulated DLQ handler. In a real scenario, this would integrate with a DLQ service.
        The agent calling this skill might also pass a DLQ handler function via context.
        """
        dlq_target = context.get("dlq_target_info", "Simulated DLQ")
        logger.warning(f"DLQ_SIM: Message ID '{message_content.get('id', 'N/A')}' sent to {dlq_target}. Reason: {reason}")
        # This is where you might call an external DLQ service or a callback from context.
        # For now, it just logs. The skill's output will indicate DLQ dispatch.

    def _default_simulated_processing_function(self, message_payload: dict, context: dict):
        """
        Default (simulated) processing function if not provided in context.
        This function simulates successes, transient errors, and permanent errors.
        It can be overridden by a 'custom_processing_function' in the context.
        """
        # Allow forcing failure for testing purposes, if specified in payload
        if 'force_failure_type' in message_payload:
            failure_type = message_payload['force_failure_type']
            if failure_type == 'transient':
                logger.info(f"Simulating forced transient error for message: {message_payload.get('id','N/A')}")
                return {'status': 'failure', 'error_type': 'transient', 'message': 'Simulated forced transient error'}
            elif failure_type == 'permanent':
                logger.info(f"Simulating forced permanent error for message: {message_payload.get('id','N/A')}")
                return {'status': 'failure', 'error_type': 'permanent', 'message': 'Simulated forced permanent error'}

        # Regular random simulation
        rand_val = random.random()
        if rand_val < 0.15: # 15% chance of permanent failure
            return {'status': 'failure', 'error_type': 'permanent', 'message': 'Simulated permanent error (e.g., invalid data structure)'}
        elif rand_val < 0.50: # 35% chance of transient failure (15% to 50%)
            return {'status': 'failure', 'error_type': 'transient', 'message': 'Simulated transient error (e.g., temporary service unavailability)'}
        else: # 50% chance of success
            return {'status': 'success', 'result': f"Successfully processed: {message_payload.get('data', 'N/A')}"}

    def process_message(self, parameters: dict, context: dict) -> dict:
        """
        Processes a message with retry logic and DLQ integration.
        Matches the 'handler_class_or_function' for 'robust_message_handler' skill.

        Args:
            parameters (dict): Input parameters. Expected:
                               - 'message_json': JSON string of the message.
                               - 'max_retries' (optional, int): Overrides skill default.
                               - 'base_retry_delay_seconds' (optional, float): Overrides skill default.
            context (dict): Context data. Expected (optional):
                            - 'custom_processing_function': A callable that takes (message_payload, context) and returns processing result.
                            - 'custom_dlq_handler': A callable that takes (message_content, reason, context).
                            - 'dlq_target_info': Information about where DLQ messages go (e.g. queue name).

        Returns:
            dict: An MCP-compliant response.
        """
        logger.info(f"RobustMCP.process_message invoked with parameters: {parameters}")

        message_json = parameters.get('message_json')
        if not message_json:
            return {"status": "error", "error_message": "Missing 'message_json' in parameters."}

        try:
            message_content = json.loads(message_json)
            message_id = message_content.get('id', f"unknown_{random.randint(1000,9999)}")
            payload = message_content.get('payload', {})
        except json.JSONDecodeError as e:
            err_msg = f"Error decoding message_json: {str(e)}"
            logger.error(err_msg)
            # Try to send the raw malformed message to DLQ if possible
            self._default_simulated_dlq_handler({"raw_message": message_json, "id": "malformed"}, err_msg, context)
            return {"status": "error", "error_message": err_msg, "data": {"action_taken": "sent_to_dlq_due_to_decode_error"}}

        # Get skill configurations, allowing overrides from parameters
        max_retries = parameters.get('max_retries', self.skill_config.get('default_max_retries', 3))
        base_delay = parameters.get('base_retry_delay_seconds', self.skill_config.get('default_base_retry_delay_seconds', 1.0))

        # Get handlers from context or use defaults
        processing_function = context.get('custom_processing_function', self._default_simulated_processing_function)
        dlq_handler = context.get('custom_dlq_handler', self._default_simulated_dlq_handler)

        logger.info(f"Processing message ID: {message_id}, Payload: {payload}. Max retries: {max_retries}, Base delay: {base_delay}s")

        for attempt in range(max_retries + 1): # +1 for the initial attempt
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for message ID: {message_id}")
            try:
                # The processing function itself might need the broader context too
                result = processing_function(payload, context)

                if result.get('status') == 'success':
                    success_msg = f"Message ID '{message_id}' processed successfully."
                    logger.info(success_msg)
                    return {"status": "success", "data": {"message_id": message_id, "result": result.get('result', 'Success'), "attempts": attempt + 1}}

                error_type = result.get('error_type', 'unknown_error_type')
                error_message = result.get('message', 'No error message provided by processor.')

                if error_type == 'permanent':
                    dlq_reason = f"Permanent failure after {attempt + 1} attempt(s): {error_message}"
                    logger.warning(f"Message ID '{message_id}' encountered permanent failure. Error: {error_message}. Sending to DLQ.")
                    dlq_handler(message_content, dlq_reason, context)
                    return {"status": "error", "error_message": dlq_reason, "data": {"message_id": message_id, "action_taken": "sent_to_dlq_permanent_failure"}}

                # Transient failure, will retry if attempts left
                logger.warning(f"Message ID '{message_id}' encountered transient failure. Error: {error_message}.")
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) # Exponential backoff
                    jitter = random.uniform(0, delay * 0.1) # Add jitter (10%)
                    actual_delay = min(delay + jitter, 60.0) # Cap delay to e.g. 60s
                    logger.info(f"Retrying message ID '{message_id}' in {actual_delay:.2f} seconds...")
                    time.sleep(actual_delay) # In a real async skill, this would be proper non-blocking delay/reschedule
                else: # Max retries reached for transient error
                    dlq_reason = f"Max retries ({max_retries + 1}) reached for message ID '{message_id}'. Last error: {error_message}"
                    logger.error(dlq_reason + ". Sending to DLQ.")
                    dlq_handler(message_content, dlq_reason, context)
                    return {"status": "error", "error_message": dlq_reason, "data": {"message_id": message_id, "action_taken": "sent_to_dlq_max_retries"}}

            except Exception as e: # Catch unexpected errors in processing_function or logic here
                critical_error_message = f"Critical unexpected error during processing of message ID '{message_id}': {str(e)}"
                logger.exception(critical_error_message) # Log with stack trace
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    actual_delay = min(delay + random.uniform(0, delay * 0.1), 60.0)
                    logger.info(f"Retrying message ID '{message_id}' in {actual_delay:.2f} seconds due to critical error...")
                    time.sleep(actual_delay)
                else: # Max retries reached after critical error
                    dlq_reason = f"Max retries ({max_retries + 1}) reached for message ID '{message_id}' after critical error. Last error: {critical_error_message}"
                    logger.error(dlq_reason + ". Sending to DLQ.")
                    dlq_handler(message_content, dlq_reason, context)
                    return {"status": "error", "error_message": dlq_reason, "data": {"message_id": message_id, "action_taken": "sent_to_dlq_critical_error"}}

        # Fallback, though theoretically unreachable if logic is correct
        final_error_msg = f"Message ID '{message_id}' exhausted processing loop unexpectedly."
        logger.error(final_error_msg)
        return {"status": "error", "error_message": final_error_msg, "data": {"message_id": message_id, "action_taken": "loop_exhausted_unexpectedly"}}


if __name__ == '__main__':
    # Example Usage (for testing the skill directly)
    skill_handler = RobustMCP(skill_config={
        'default_max_retries': 2,
        'default_base_retry_delay_seconds': 0.1
    })
    test_context = {"dlq_target_info": "TestDLQ"}

    print("\n--- Test 1: Successful Processing ---")
    params1 = {"message_json": json.dumps({"id": "msg_success", "payload": {"data": "important_payload"}})}
    response1 = skill_handler.process_message(parameters=params1, context=test_context)
    print(json.dumps(response1, indent=2))
    assert response1["status"] == "success"

    print("\n--- Test 2: Permanent Failure ---")
    params2 = {"message_json": json.dumps({"id": "msg_permanent", "payload": {"force_failure_type": "permanent"}})}
    response2 = skill_handler.process_message(parameters=params2, context=test_context)
    print(json.dumps(response2, indent=2))
    assert response2["status"] == "error"
    assert response2["data"]["action_taken"] == "sent_to_dlq_permanent_failure"

    print("\n--- Test 3: Transient Failure then Success (with default processor randomness) ---")
    # This test is probabilistic with the default processor. For deterministic, mock/pass custom_processing_function.
    # To make it more likely to pass through retry:
    # We can't directly control the default _default_simulated_processing_function's randomness from here easily
    # So this test might sometimes succeed on first try, or go to DLQ.
    # For a true unit test of retry, mock the processing_function.
    # Here, we just run it to see an example path.
    params3 = {"message_json": json.dumps({"id": "msg_transient_maybe", "payload": {"data": "flaky data"}})}
    # Override max_retries for this call to see more attempts
    params3_with_more_retries = {**params1, "max_retries": 3}
    # response3 = skill_handler.process_message(parameters=params3_with_more_retries, context=test_context)
    # print(json.dumps(response3, indent=2))
    # The assertion here would be tricky due to randomness.

    print("\n--- Test 4: Max Retries leading to DLQ (forced transient) ---")
    params4 = {"message_json": json.dumps({"id": "msg_max_retry", "payload": {"force_failure_type": "transient"}})}
    # Use skill_config default_max_retries = 2 (so 3 attempts total)
    response4 = skill_handler.process_message(parameters=params4, context=test_context) # Will use default 2 retries
    print(json.dumps(response4, indent=2))
    assert response4["status"] == "error"
    assert response4["data"]["action_taken"] == "sent_to_dlq_max_retries"

    print("\n--- Test 5: Malformed JSON ---")
    params5 = {"message_json": "{\"id\":\"test_malformed\""} # Malformed
    response5 = skill_handler.process_message(parameters=params5, context=test_context)
    print(json.dumps(response5, indent=2))
    assert response5["status"] == "error"
    assert "Error decoding message_json" in response5["error_message"]
    assert response5["data"]["action_taken"] == "sent_to_dlq_due_to_decode_error"

    print("\n--- Test 6: Custom Processing Function (Success) ---")
    def my_always_succeed_processor(payload, context):
        return {'status': 'success', 'result': f"Custom processed: {payload.get('custom_val')}"}
    custom_context_succ = {**test_context, "custom_processing_function": my_always_succeed_processor}
    params6 = {"message_json": json.dumps({"id": "msg_custom_succ", "payload": {"custom_val": "abc"}})}
    response6 = skill_handler.process_message(parameters=params6, context=custom_context_succ)
    print(json.dumps(response6, indent=2))
    assert response6["status"] == "success"
    assert "Custom processed: abc" in response6["data"]["result"]

    print("\n--- Test 7: Custom Processing Function (Permanent Failure) ---")
    def my_always_fail_perm_processor(payload, context):
        return {'status': 'failure', 'error_type': 'permanent', 'message': "Custom permanent fail"}
    custom_context_perm_fail = {**test_context, "custom_processing_function": my_always_fail_perm_processor}
    params7 = {"message_json": json.dumps({"id": "msg_custom_perm", "payload": {}})}
    response7 = skill_handler.process_message(parameters=params7, context=custom_context_perm_fail)
    print(json.dumps(response7, indent=2))
    assert response7["status"] == "error"
    assert response7["data"]["action_taken"] == "sent_to_dlq_permanent_failure"
    assert "Custom permanent fail" in response7["error_message"]

    logger.info("Robust messaging skill direct tests completed.")

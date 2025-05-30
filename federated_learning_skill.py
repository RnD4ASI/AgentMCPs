import json
import random
import logging

# Setup basic logging for the skill
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FederatedLearningMCP: # Matching class name from mcp_configurations.json
    """
    MCP Skill for simulating a federated learning process.
    Matches 'federated_model_aggregator' skill in mcp_configurations.json.
    The skill actions are dispatched by the 'execute_action' method based on 'action' parameter.
    This skill is more complex as it implies state management (global model, round updates)
    which typically would involve external storage or be managed by the calling agent via context.
    For this refactor, we'll simulate some aspects in-memory if context doesn't provide them,
    or expect them from context.
    """

    def __init__(self, skill_config=None):
        """
        Initializes the FederatedLearningSkill.

        Args:
            skill_config (dict, optional): Configuration specific to the skill's instantiation.
                                           Example: {'default_model_id': 'my_model_v1'}
        """
        self.skill_config = skill_config if skill_config else {}
        logger.info("FederatedLearningMCP skill initialized.")
        # Note: True state (global_model_parameters, participant_updates, current_round)
        # should ideally be passed via `context` for stateless skill execution or managed
        # by an orchestrator. This implementation will try to read from context first.

    def _get_state_from_context(self, context, key, default_value):
        """Helper to retrieve stateful information from context, or use a default."""
        if context and key in context:
            return context[key]
        logger.warning(f"'{key}' not found in context. Using default or initial value: {default_value}")
        return default_value

    def _initialize_global_model(self):
        """Simulates the initialization of global model parameters if not in context."""
        logger.info("Initializing new global model parameters (simulation).")
        return {"weights": [random.uniform(-0.1, 0.1) for _ in range(5)], "bias": random.uniform(-0.05, 0.05)}

    def execute_action(self, parameters: dict, context: dict) -> dict:
        """
        Executes a specific federated learning action.
        This method acts as the main entry point for the skill.

        Args:
            parameters (dict): Input parameters. Expected:
                               - 'action': (str) The FL action to perform.
                               - 'round_id': (int, optional)
                               - 'participant_id': (str, optional)
                               - 'model_update_json': (str, optional JSON)
                               - 'aggregation_strategy': (str, optional, default 'average')
                               - 'global_model_id': (str, optional)
            context (dict): Context data. Expected to potentially contain:
                            - 'global_model_parameters': (dict) The current global model.
                            - 'participant_updates_for_round': (dict) Updates for the current round.
                            - 'current_round_id': (int) The active round ID.
        Returns:
            dict: An MCP-compliant response.
        """
        action = parameters.get('action')
        if not action:
            return {"status": "error", "error_message": "Missing 'action' in parameters."}

        logger.info(f"FederatedLearningMCP executing action: {action} with params: {parameters}")

        # Retrieve stateful components from context or initialize them.
        # This is a simplified state management for the skill.
        # A robust MCP skill would rely on the calling agent/orchestrator to manage state via context.
        global_model_id = parameters.get('global_model_id', self.skill_config.get('default_model_id', 'default_fl_model'))
        current_round_id = self._get_state_from_context(context, 'current_round_id', 0)
        global_model_parameters = self._get_state_from_context(context, 'global_model_parameters', None)
        if global_model_parameters is None and action not in ['get_global_model', 'start_new_round']: # Some actions might not need it pre-loaded
             global_model_parameters = self._initialize_global_model()
        elif global_model_parameters is None and action == 'start_new_round': # Initialize if starting and none exists
             global_model_parameters = self._initialize_global_model()


        # Participant updates should be scoped by round_id in context if passed fully.
        # e.g., context: {'participant_updates': {1: {'pA': updateA, 'pB': updateB}}}
        # For simplicity, if 'participant_updates_for_round' is passed, it's for the 'current_round_id'.
        participant_updates_for_round = self._get_state_from_context(context, f'participant_updates_round_{current_round_id}', {})


        if action == "start_new_round":
            round_id = parameters.get('round_id')
            if not isinstance(round_id, int) or round_id <= current_round_id :
                return {"status": "error", "error_message": f"Invalid or old round_id: {round_id}. Current round is {current_round_id}."}

            # State update simulation: new round started.
            # The response should inform the agent what to store in context for the next call.
            new_context_state = {
                'current_round_id': round_id,
                'global_model_parameters': global_model_parameters, # Return current model for participants
                f'participant_updates_round_{round_id}': {} # Initialize empty updates for new round
            }
            return {
                "status": "success",
                "data": {
                    "message": f"Round {round_id} started for model '{global_model_id}'.",
                    "global_model_parameters": global_model_parameters,
                    "current_round_id": round_id
                },
                "context_updates_suggestion": new_context_state # Suggests what agent might want to update in its context store
            }

        elif action == "submit_model_update":
            round_id = parameters.get('round_id')
            participant_id = parameters.get('participant_id')
            model_update_json = parameters.get('model_update_json')

            if not all([isinstance(round_id, int), participant_id, model_update_json]):
                return {"status": "error", "error_message": "Missing round_id, participant_id, or model_update_json for submit_model_update."}
            if round_id != current_round_id:
                return {"status": "error", "error_message": f"Update for round {round_id} rejected. Current active round is {current_round_id}."}

            try:
                model_update = json.loads(model_update_json)
                if not all(key in model_update for key in ['weights_delta', 'bias_delta', 'data_samples_count']):
                    return {"status": "error", "error_message": "Invalid model update structure. Required keys: 'weights_delta', 'bias_delta', 'data_samples_count'."}

                # Update participant_updates_for_round (which is a copy or new dict)
                participant_updates_for_round[participant_id] = model_update

                # Suggest context update for this round's submissions
                context_update_key = f'participant_updates_round_{current_round_id}'
                new_context_state_suggestion = {context_update_key: participant_updates_for_round}

                return {
                    "status": "success",
                    "data": {"message": f"Participant {participant_id}'s update for round {round_id} received. Samples: {model_update['data_samples_count']}"},
                    "context_updates_suggestion": new_context_state_suggestion
                }
            except json.JSONDecodeError as e:
                return {"status": "error", "error_message": f"Error decoding model_update_json: {str(e)}"}
            except Exception as e:
                 return {"status": "error", "error_message": f"Error processing model update: {str(e)}"}


        elif action == "aggregate_updates":
            round_id = parameters.get('round_id')
            if not isinstance(round_id, int) or round_id != current_round_id:
                return {"status": "error", "error_message": f"Cannot aggregate for round {round_id}. Current active round is {current_round_id}."}
            if not global_model_parameters:
                 return {"status": "error", "error_message": "Global model parameters not available for aggregation."}

            updates_to_aggregate = participant_updates_for_round
            if not updates_to_aggregate:
                return {"status": "success", "data": {"message": f"No updates received for round {round_id}. Aggregation skipped.", "global_model_parameters": global_model_parameters}}

            aggregation_strategy = parameters.get('aggregation_strategy', 'average')
            num_model_weights = len(global_model_parameters.get("weights", []))

            new_weights = list(global_model_parameters["weights"]) # Make a copy
            new_bias = global_model_parameters["bias"]

            if aggregation_strategy == 'average':
                valid_updates_count = 0
                aggregated_weights_delta = [0.0] * num_model_weights
                aggregated_bias_delta = 0.0
                for update in updates_to_aggregate.values():
                    if len(update["weights_delta"]) == num_model_weights:
                        for i in range(num_model_weights):
                            aggregated_weights_delta[i] += update["weights_delta"][i]
                        aggregated_bias_delta += update["bias_delta"]
                        valid_updates_count += 1
                if valid_updates_count > 0:
                    for i in range(num_model_weights):
                        new_weights[i] += aggregated_weights_delta[i] / valid_updates_count
                    new_bias += aggregated_bias_delta / valid_updates_count

            elif aggregation_strategy == 'weighted_average':
                total_samples = 0
                aggregated_weights_delta = [0.0] * num_model_weights
                aggregated_bias_delta = 0.0
                for update in updates_to_aggregate.values():
                    if len(update["weights_delta"]) == num_model_weights and update["data_samples_count"] > 0:
                        weight_factor = update["data_samples_count"]
                        for i in range(num_model_weights):
                            aggregated_weights_delta[i] += update["weights_delta"][i] * weight_factor
                        aggregated_bias_delta += update["bias_delta"] * weight_factor
                        total_samples += weight_factor
                if total_samples > 0:
                    for i in range(num_model_weights):
                        new_weights[i] += aggregated_weights_delta[i] / total_samples
                    new_bias += aggregated_bias_delta / total_samples
            else:
                return {"status": "error", "error_message": f"Unknown aggregation strategy: {aggregation_strategy}"}

            updated_global_model = {"weights": new_weights, "bias": new_bias}
            # Suggest context update for the global model
            new_context_state_suggestion = {'global_model_parameters': updated_global_model}
            return {
                "status": "success",
                "data": {
                    "message": f"Global model updated for round {round_id} using '{aggregation_strategy}'.",
                    "global_model_parameters": updated_global_model
                },
                "context_updates_suggestion": new_context_state_suggestion
            }

        elif action == "get_global_model":
            if not global_model_parameters: # If it wasn't loaded from context and not initialized yet
                global_model_parameters = self._initialize_global_model() # Initialize a default one
            return {"status": "success", "data": {"global_model_parameters": global_model_parameters, "current_round_id": current_round_id}}

        else:
            return {"status": "error", "error_message": f"Unknown action: {action}"}


if __name__ == '__main__':
    # Example Usage (for testing the skill directly)
    # This simulates how an agent might interact with the skill, managing context.
    skill_handler = FederatedLearningMCP(skill_config={'default_model_id': 'example_model'})

    # --- Round 1 ---
    print("--- Starting Round 1 ---")
    start_round_params = {"action": "start_new_round", "round_id": 1}
    # Context would be empty or contain previous state. For first round, it's empty.
    current_context = {}
    response_start_round = skill_handler.execute_action(start_round_params, current_context)
    print(json.dumps(response_start_round, indent=2))
    assert response_start_round["status"] == "success"
    # Agent updates its context based on 'context_updates_suggestion' or the data part.
    current_context.update(response_start_round.get("context_updates_suggestion", {}))
    # Or more explicitly:
    # current_context['current_round_id'] = response_start_round['data']['current_round_id']
    # current_context['global_model_parameters'] = response_start_round['data']['global_model_parameters']
    # current_context[f"participant_updates_round_{current_context['current_round_id']}"] = {}


    print("\n--- Submitting Updates for Round 1 ---")
    update_p1_r1 = {"weights_delta": [0.1, -0.05, 0.02, 0.0, -0.1], "bias_delta": 0.01, "data_samples_count": 100}
    submit_p1_params = {
        "action": "submit_model_update",
        "round_id": 1,
        "participant_id": "participant_A",
        "model_update_json": json.dumps(update_p1_r1)
    }
    response_submit_p1 = skill_handler.execute_action(submit_p1_params, current_context)
    print(json.dumps(response_submit_p1, indent=2))
    assert response_submit_p1["status"] == "success"
    current_context.update(response_submit_p1.get("context_updates_suggestion", {}))


    update_p2_r1 = {"weights_delta": [0.05, 0.0, -0.03, 0.01, -0.15], "bias_delta": -0.02, "data_samples_count": 150}
    submit_p2_params = {
        "action": "submit_model_update",
        "round_id": 1,
        "participant_id": "participant_B",
        "model_update_json": json.dumps(update_p2_r1)
    }
    response_submit_p2 = skill_handler.execute_action(submit_p2_params, current_context)
    print(json.dumps(response_submit_p2, indent=2))
    assert response_submit_p2["status"] == "success"
    current_context.update(response_submit_p2.get("context_updates_suggestion", {}))

    print(f"\nContext after submissions for round 1: {current_context[f'participant_updates_round_{current_context['current_round_id']}']}")

    print("\n--- Aggregating Updates for Round 1 ---")
    aggregate_r1_params = {"action": "aggregate_updates", "round_id": 1, "aggregation_strategy": "weighted_average"}
    response_aggregate_r1 = skill_handler.execute_action(aggregate_r1_params, current_context)
    print(json.dumps(response_aggregate_r1, indent=2))
    assert response_aggregate_r1["status"] == "success"
    current_context.update(response_aggregate_r1.get("context_updates_suggestion", {}))
    print(f"Updated global model (in context): {current_context['global_model_parameters']}")

    # --- Round 2 ---
    print("\n--- Starting Round 2 ---")
    start_round2_params = {"action": "start_new_round", "round_id": 2}
    response_start_round2 = skill_handler.execute_action(start_round2_params, current_context)
    print(json.dumps(response_start_round2, indent=2))
    assert response_start_round2["status"] == "success"
    current_context.update(response_start_round2.get("context_updates_suggestion", {}))
    # Note: participant_updates_round_1 would still be in context unless explicitly cleared by agent.
    # The skill now initializes participant_updates_round_2.

    logger.info("Federated learning skill direct tests completed.")

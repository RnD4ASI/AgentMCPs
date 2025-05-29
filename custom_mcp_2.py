import json
import random

class FederatedLearningMCP:
    """
    Conceptual MCP to simulate a federated learning process.
    This MCP coordinates the aggregation of model updates from multiple
    participants without accessing their raw data.
    """

    def __init__(self, global_model_id, aggregation_strategy='average'):
        """
        Initializes the FederatedLearningMCP.

        Args:
            global_model_id (str): Identifier for the global model being trained.
            aggregation_strategy (str): The strategy to use for aggregating model updates
                                        (e.g., 'average', 'weighted_average').
        """
        self.global_model_id = global_model_id
        self.aggregation_strategy = aggregation_strategy
        self.current_round = 0
        self.participant_updates = {} # Stores updates for the current round: {round_id: {participant_id: update}}
        self.global_model_parameters = self._initialize_global_model() # Simulate initial global model params
        
        print(f"FederatedLearningMCP initialized for model '{self.global_model_id}'. Strategy: {self.aggregation_strategy}")
        print(f"Initial global model parameters: {self.global_model_parameters}")

    def _initialize_global_model(self):
        """Simulates the initialization of global model parameters."""
        # In a real scenario, this would be the actual model's initial state
        return {"weights": [random.uniform(-1, 1) for _ in range(5)], "bias": random.uniform(-0.5, 0.5)}

    def start_new_round(self, round_id):
        """
        Starts a new round of federated learning.

        Args:
            round_id (int): The identifier for the new round.
        """
        if round_id <= self.current_round and self.current_round != 0 :
            print(f"Error: Round {round_id} has already been processed or is not newer than current round {self.current_round}.")
            return
            
        self.current_round = round_id
        self.participant_updates[self.current_round] = {}
        print(f"\n--- Round {self.current_round} started for model '{self.global_model_id}' ---")
        # In a real system, the MCP would now notify participants to start local training
        # and provide them with the current global_model_parameters.
        return self.global_model_parameters # Participants would fetch this

    def submit_model_update(self, participant_id, round_id, model_update_json):
        """
        Allows a participant to submit their model update for a specific round.

        Args:
            participant_id (str): Unique identifier for the participant.
            round_id (int): The round for which this update is submitted.
            model_update_json (str): JSON string containing the model update
                                     (e.g., gradients or updated weights).
                                     Example: '{"weights_delta": [...], "bias_delta": ...}'
        """
        if round_id != self.current_round:
            print(f"Warning: Participant {participant_id} submitted update for round {round_id}, but current round is {self.current_round}. Update ignored.")
            return

        if participant_id in self.participant_updates.get(self.current_round, {}):
            print(f"Warning: Participant {participant_id} already submitted an update for round {self.current_round}. Overwriting.")

        try:
            model_update = json.loads(model_update_json)
            # Basic validation of update structure (can be more sophisticated)
            if not isinstance(model_update, dict) or \
               not all(key in model_update for key in ['weights_delta', 'bias_delta', 'data_samples_count']):
                print(f"Error: Invalid model update structure from {participant_id}. Required keys: 'weights_delta', 'bias_delta', 'data_samples_count'.")
                return

            self.participant_updates[self.current_round][participant_id] = model_update
            print(f"Participant {participant_id} submitted update for round {self.current_round}. Samples: {model_update['data_samples_count']}")
        except json.JSONDecodeError as e:
            print(f"Error decoding model update from {participant_id}: {e}")
        except Exception as e:
            print(f"Error processing model update from {participant_id}: {e}")


    def aggregate_updates(self, round_id):
        """
        Aggregates all received model updates for the specified round and
        updates the global model.

        Args:
            round_id (int): The round for which to aggregate updates.
        
        Returns:
            dict: The new global model parameters after aggregation, or None if aggregation failed.
        """
        if round_id != self.current_round:
            print(f"Error: Cannot aggregate for round {round_id} as it's not the current round.")
            return None
        
        updates_for_round = self.participant_updates.get(round_id, {})
        if not updates_for_round:
            print(f"No updates received for round {round_id}. Aggregation skipped.")
            return self.global_model_parameters # Return current params

        print(f"Aggregating updates for round {round_id} from {len(updates_for_round)} participants.")

        if self.aggregation_strategy == 'average':
            # Simple averaging of weight deltas and bias delta
            num_participants = len(updates_for_round)
            aggregated_weights_delta = [0.0] * len(self.global_model_parameters["weights"])
            aggregated_bias_delta = 0.0
            total_samples = 0

            for participant_id, update in updates_for_round.items():
                if len(update["weights_delta"]) != len(aggregated_weights_delta):
                    print(f"Error: Mismatch in weights dimension from {participant_id}. Skipping their update.")
                    num_participants -=1 # Adjust count for averaging
                    continue
                for i in range(len(update["weights_delta"])):
                    aggregated_weights_delta[i] += update["weights_delta"][i]
                aggregated_bias_delta += update["bias_delta"]
                total_samples += update["data_samples_count"]
            
            if num_participants > 0:
                # Update global model by applying averaged deltas
                for i in range(len(self.global_model_parameters["weights"])):
                    self.global_model_parameters["weights"][i] += aggregated_weights_delta[i] / num_participants
                self.global_model_parameters["bias"] += aggregated_bias_delta / num_participants
                print(f"Global model updated for round {round_id} using 'average' strategy on {num_participants} updates. Total samples considered: {total_samples}.")
            else:
                print(f"No valid updates to aggregate for round {round_id}.")

        elif self.aggregation_strategy == 'weighted_average':
            # Weighted averaging based on data_samples_count
            aggregated_weights_delta = [0.0] * len(self.global_model_parameters["weights"])
            aggregated_bias_delta = 0.0
            total_samples = 0
            valid_updates_count = 0

            for participant_id, update in updates_for_round.items():
                if len(update["weights_delta"]) != len(aggregated_weights_delta) or update["data_samples_count"] <= 0:
                    print(f"Error: Mismatch in weights dimension or invalid sample count from {participant_id}. Skipping their update.")
                    continue
                
                weight_factor = update["data_samples_count"]
                for i in range(len(update["weights_delta"])):
                    aggregated_weights_delta[i] += update["weights_delta"][i] * weight_factor
                aggregated_bias_delta += update["bias_delta"] * weight_factor
                total_samples += weight_factor
                valid_updates_count +=1
            
            if total_samples > 0 and valid_updates_count > 0:
                for i in range(len(self.global_model_parameters["weights"])):
                    self.global_model_parameters["weights"][i] += aggregated_weights_delta[i] / total_samples
                self.global_model_parameters["bias"] += aggregated_bias_delta / total_samples
                print(f"Global model updated for round {round_id} using 'weighted_average' strategy on {valid_updates_count} updates. Total sample weight: {total_samples}.")
            else:
                print(f"No valid updates with positive sample counts to aggregate for round {round_id}.")
        else:
            print(f"Error: Unknown aggregation strategy '{self.aggregation_strategy}'.")
            return None
        
        print(f"New global model parameters: {self.global_model_parameters}")
        # In a real system, this updated model would be available for participants for the next round.
        return self.global_model_parameters

    def get_global_model(self):
        """Returns the current global model parameters."""
        return self.global_model_parameters

if __name__ == '__main__':
    print("--- FederatedLearningMCP Demo ---")
    
    # Initialize MCP with weighted average strategy
    fl_mcp = FederatedLearningMCP(global_model_id="image_classifier_v1", aggregation_strategy='weighted_average')

    # --- Round 1 ---
    current_model_params_round_1 = fl_mcp.start_new_round(round_id=1)
    # Participants would use current_model_params_round_1 for local training

    # Simulate participants submitting updates
    update_p1_r1 = {"weights_delta": [0.1, -0.05, 0.02, 0.0, -0.1], "bias_delta": 0.01, "data_samples_count": 100}
    fl_mcp.submit_model_update(participant_id="participant_A", round_id=1, model_update_json=json.dumps(update_p1_r1))

    update_p2_r1 = {"weights_delta": [0.05, 0.0, -0.03, 0.01, -0.15], "bias_delta": -0.02, "data_samples_count": 150}
    fl_mcp.submit_model_update(participant_id="participant_B", round_id=1, model_update_json=json.dumps(update_p2_r1))
    
    update_p3_r1_invalid_dim = {"weights_delta": [0.05], "bias_delta": -0.02, "data_samples_count": 50} # Invalid dimensions
    fl_mcp.submit_model_update(participant_id="participant_C", round_id=1, model_update_json=json.dumps(update_p3_r1_invalid_dim))


    # Aggregate updates for Round 1
    new_global_model_r1 = fl_mcp.aggregate_updates(round_id=1)
    if new_global_model_r1:
        print(f"Global model after Round 1: {new_global_model_r1}")

    # --- Round 2 ---
    current_model_params_round_2 = fl_mcp.start_new_round(round_id=2)

    update_p1_r2 = {"weights_delta": [0.08, -0.02, 0.01, 0.03, -0.05], "bias_delta": 0.005, "data_samples_count": 120}
    fl_mcp.submit_model_update(participant_id="participant_A", round_id=2, model_update_json=json.dumps(update_p1_r2))

    # Participant B submits for wrong round (should be ignored for aggregation)
    update_p2_r1_late = {"weights_delta": [0.1, 0.1, -0.1, 0.1, -0.1], "bias_delta": -0.01, "data_samples_count": 20}
    fl_mcp.submit_model_update(participant_id="participant_B", round_id=1, model_update_json=json.dumps(update_p2_r1_late))

    update_pD_r2 = {"weights_delta": [0.02, 0.01, -0.01, 0.05, -0.08], "bias_delta": -0.01, "data_samples_count": 80}
    fl_mcp.submit_model_update(participant_id="participant_D", round_id=2, model_update_json=json.dumps(update_pD_r2))

    # Aggregate updates for Round 2
    new_global_model_r2 = fl_mcp.aggregate_updates(round_id=2)
    if new_global_model_r2:
        print(f"Global model after Round 2: {new_global_model_r2}")
        
    print("\n--- End of FederatedLearningMCP Demo ---")

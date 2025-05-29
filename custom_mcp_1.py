import time
import json
import random

class RealTimeStreamingMCP:
    """
    MCP designed for handling real-time data streams with dynamic
    validation and transformation.
    """

    def __init__(self, config):
        """
        Initializes the RealTimeStreamingMCP.

        Args:
            config (dict): Configuration for the MCP, e.g., validation rules,
                           transformation logic.
                           Expected keys:
                           - 'validation_rules': dict of field -> {type, min, max}
                           - 'transformation_map': dict of field -> new_field_name or lambda func
        """
        self.config = config
        self.data_callbacks = []
        print(f"RealTimeStreamingMCP initialized with config: {config}")

    def register_data_callback(self, callback):
        """
        Registers a callback function to be invoked with processed data.

        Args:
            callback (function): A function that takes processed data as an argument.
        """
        self.data_callbacks.append(callback)
        print(f"Callback {callback.__name__} registered.")

    def _validate_data(self, data_item):
        """
        Validates a single data item against rules in self.config['validation_rules'].

        Args:
            data_item (dict): The data item to validate.

        Returns:
            bool: True if valid, False otherwise.
            list: List of validation errors.
        """
        errors = []
        rules = self.config.get('validation_rules', {})
        for field, rule in rules.items():
            if field not in data_item:
                errors.append(f"Missing field: {field}")
                continue

            value = data_item[field]
            if 'type' in rule and not isinstance(value, rule['type']):
                errors.append(f"Invalid type for {field}: expected {rule['type']}, got {type(value)}")
            if 'min' in rule and value < rule['min']:
                errors.append(f"Value for {field} ({value}) is less than min ({rule['min']})")
            if 'max' in rule and value > rule['max']:
                errors.append(f"Value for {field} ({value}) is greater than max ({rule['max']})")
        
        if errors:
            print(f"Validation failed for {data_item}: {errors}")
            return False, errors
        return True, []

    def _transform_data(self, data_item):
        """
        Transforms a single data item based on self.config['transformation_map'].

        Args:
            data_item (dict): The data item to transform.

        Returns:
            dict: The transformed data item.
        """
        transformed_item = data_item.copy()
        transform_map = self.config.get('transformation_map', {})
        
        # Example: Convert temperature from Celsius to Fahrenheit
        if 'temperature_celsius' in transformed_item and 'temperature_celsius' in transform_map:
            celsius = transformed_item['temperature_celsius']
            if isinstance(celsius, (int, float)):
                 # Apply transformation logic defined in config, e.g., a lambda
                if callable(transform_map['temperature_celsius']):
                    transformed_item['temperature_fahrenheit'] = transform_map['temperature_celsius'](celsius)
                    del transformed_item['temperature_celsius'] # remove old key if desired
                else: # or a simple rename
                    new_key = transform_map['temperature_celsius']
                    transformed_item[new_key] = (celsius * 9/5) + 32 
                    if new_key != 'temperature_celsius':
                         del transformed_item['temperature_celsius']

        # Example: Add a processing timestamp
        if 'add_timestamp' in transform_map and transform_map['add_timestamp']:
            transformed_item['processing_timestamp_utc'] = time.time()
            
        print(f"Data transformed from {data_item} to {transformed_item}")
        return transformed_item

    def process_stream_data(self, raw_data_json):
        """
        Processes a raw data string (assumed JSON list of items),
        validates, transforms, and then invokes registered callbacks.

        Args:
            raw_data_json (str): A JSON string representing a list of data items.
        """
        print(f"\nReceived raw data: {raw_data_json}")
        try:
            data_list = json.loads(raw_data_json)
            if not isinstance(data_list, list):
                print("Error: Input data must be a JSON list of objects.")
                return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

        processed_count = 0
        for item in data_list:
            is_valid, errors = self._validate_data(item)
            if is_valid:
                transformed_item = self._transform_data(item)
                for callback in self.data_callbacks:
                    try:
                        callback(transformed_item)
                    except Exception as e:
                        print(f"Error in callback {callback.__name__}: {e}")
                processed_count += 1
            else:
                print(f"Skipping invalid item: {item}. Errors: {errors}")
        print(f"Finished processing stream. Successfully processed {processed_count}/{len(data_list)} items.")


def example_data_consumer(processed_data):
    """Example callback function to consume processed data."""
    print(f"Data Consumer: Received processed data: {processed_data}")

if __name__ == '__main__':
    print("--- RealTimeStreamingMCP Demo ---")
    
    # Configuration for the MCP
    mcp_config = {
        'validation_rules': {
            'sensor_id': {'type': str},
            'timestamp': {'type': int, 'min': 1600000000},
            'temperature_celsius': {'type': float, 'min': -50.0, 'max': 100.0},
            'humidity_percent': {'type': float, 'min': 0.0, 'max': 100.0}
        },
        'transformation_map': {
            'temperature_celsius': lambda c: (c * 9/5) + 32, # Convert C to F and rename later
            'add_timestamp': True
        }
    }

    streaming_mcp = RealTimeStreamingMCP(config=mcp_config)
    streaming_mcp.register_data_callback(example_data_consumer)

    # Simulate receiving a stream of data
    raw_stream_data_1 = """
    [
        {"sensor_id": "A101", "timestamp": 1678886400, "temperature_celsius": 22.5, "humidity_percent": 45.0},
        {"sensor_id": "B202", "timestamp": 1678886405, "temperature_celsius": -60.0, "humidity_percent": 50.0} 
    ]
    """
    streaming_mcp.process_stream_data(raw_stream_data_1)

    raw_stream_data_2 = """
    [
        {"sensor_id": "C303", "timestamp": 1678886410, "temperature_celsius": 30.1, "humidity_percent": 60.2, "pressure_hpa": 1012},
        {"sensor_id": "D404", "timestamp": 1500000000, "temperature_celsius": 25.0} 
    ]
    """
    # Note: C303 has an extra field 'pressure_hpa' (should be fine)
    # Note: D404 has old timestamp and missing humidity_percent (should fail validation)
    streaming_mcp.process_stream_data(raw_stream_data_2)
    
    raw_stream_data_3 = '[{"sensor_id": "E505", "timestamp": 1678886415, "temperature_celsius": "hot", "humidity_percent": 75.0}]' # invalid temp type
    streaming_mcp.process_stream_data(raw_stream_data_3)

    print("\n--- End of RealTimeStreamingMCP Demo ---")

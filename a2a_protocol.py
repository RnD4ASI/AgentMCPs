import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A2A Constants
A2A_VERSION = "0.2.1" # Based on the latest version seen in GitHub

class AgentCard:
    """
    Represents an A2A Agent Card, describing an agent's capabilities and connection info.
    See: https://google-a2a.github.io/A2A/specification/#agentcard
    """
    def __init__(self, agent_id: str, name: str, description: str,
                 version: str, capabilities: List[Dict[str, Any]],
                 endpoint_url: str, documentation_url: Optional[str] = None,
                 protocol_version: str = A2A_VERSION):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = version  # Agent's own version
        self.protocol_version = protocol_version # A2A protocol version
        self.capabilities = capabilities # List of skills/actions the agent can perform
        self.endpoint_url = endpoint_url # URL for A2A communication
        self.documentation_url = documentation_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "endpointUrl": self.endpoint_url,
            "documentationUrl": self.documentation_url,
            # Other fields like auth_schemes, metadata can be added as needed
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCard':
        return cls(
            agent_id=data.get("agentId", str(uuid.uuid4())), # Generate if missing
            name=data["name"],
            description=data["description"],
            version=data["version"],
            capabilities=data["capabilities"],
            endpoint_url=data["endpointUrl"],
            documentation_url=data.get("documentationUrl"),
            protocol_version=data.get("protocolVersion", A2A_VERSION)
        )

class A2ATask:
    """
    Represents an A2A Task.
    See: https://google-a2a.github.io/A2A/specification/#task
    """
    def __init__(self, task_id: str, client_agent_id: str, remote_agent_id: str,
                 capability_name: str, parameters: Dict[str, Any],
                 status: str = "pending", artifacts: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.client_agent_id = client_agent_id
        self.remote_agent_id = remote_agent_id
        self.capability_name = capability_name
        self.parameters = parameters
        self.status = status # e.g., pending, in_progress, completed, failed, cancelled
        self.artifacts = artifacts if artifacts else [] # Output of the task
        self.metadata = metadata if metadata else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.task_id,
            "clientAgentId": self.client_agent_id,
            "remoteAgentId": self.remote_agent_id,
            "capabilityName": self.capability_name,
            "parameters": self.parameters,
            "status": self.status,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2ATask':
        return cls(
            task_id=data.get("taskId", str(uuid.uuid4())),
            client_agent_id=data["clientAgentId"],
            remote_agent_id=data["remoteAgentId"],
            capability_name=data["capabilityName"],
            parameters=data["parameters"],
            status=data.get("status", "pending"),
            artifacts=data.get("artifacts"),
            metadata=data.get("metadata")
        )

class A2AClient:
    """
    Handles client-side A2A communication (sending requests to remote agents).
    This will be expanded with methods to call JSON-RPC endpoints.
    """
    def __init__(self, http_client: Optional[Any] = None):
        # In a real implementation, http_client would be something like requests.Session
        # For now, we'll mock it or use a simple HTTP client.
        try:
            import requests
            self.http_client = http_client if http_client else requests.Session()
        except ImportError:
            logging.warning("requests library not found. A2AClient will not be able to make HTTP calls.")
            self.http_client = None
        logging.info("A2AClient initialized.")

    async def send_task_request(self, remote_agent_card: AgentCard, task: A2ATask) -> Dict[str, Any]:
        """
        Sends a task request to a remote agent.
        This would typically be a JSON-RPC call to a method like 'executeTask'.
        """
        if not self.http_client:
            raise ConnectionError("HTTP client not available. Install 'requests' library.")

        endpoint_url = remote_agent_card.endpoint_url
        # This is a simplified representation. A2A uses JSON-RPC 2.0.
        # The actual payload would be a JSON-RPC request object.
        payload = {
            "jsonrpc": "2.0",
            "method": "executeTask", # Standard or custom method name
            "params": task.to_dict(),
            "id": str(uuid.uuid4())
        }

        logging.info(f"Sending A2A task request to {endpoint_url}: {json.dumps(payload, indent=2)}")

        try:
            # This is a placeholder for the actual HTTP POST request
            # In a real scenario, you'd use self.http_client.post(...)
            # and handle responses, errors, async operations correctly.
            # response = await self.http_client.post(endpoint_url, json=payload, timeout=30) # Example with httpx
            # response.raise_for_status()
            # return response.json()

            # Mocking response for now
            logging.warning("Mocking A2A task request response. No actual HTTP call made.")
            mock_response = {
                "jsonrpc": "2.0",
                "result": {"status": "received", "taskId": task.task_id, "message": "Task received by remote agent"},
                "id": payload["id"]
            }
            return mock_response

        except Exception as e:
            logging.error(f"Error sending A2A task request to {endpoint_url}: {e}")
            # Fallback error structure, or re-raise as a specific A2AError
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": payload["id"]
            }

# Further classes for A2AServer (handling incoming requests), message structures, etc., will be added.
# For now, this provides the basic building blocks.

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("A2A Protocol Module")

    # Create a sample Agent Card
    sample_capability = {
        "name": "translate_text",
        "description": "Translates text from one language to another.",
        "parameters": [
            {"name": "text", "type": "string", "required": True},
            {"name": "target_language", "type": "string", "required": True},
            {"name": "source_language", "type": "string", "required": False}
        ],
        "returns": {
            "type": "string",
            "description": "Translated text."
        }
    }

    agent_card = AgentCard(
        agent_id="agent-007",
        name="TranslationAgent",
        description="An agent that provides text translation services.",
        version="1.0.0",
        capabilities=[sample_capability],
        endpoint_url="http://localhost:8080/a2a",
        documentation_url="http://example.com/docs/translation_agent"
    )
    print("\nSample Agent Card:")
    print(agent_card.to_json())

    # Create a sample A2A Task
    task = A2ATask(
        task_id="task-123",
        client_agent_id="agent-client-001",
        remote_agent_id="agent-007",
        capability_name="translate_text",
        parameters={"text": "Hello, world!", "target_language": "es"}
    )
    print("\nSample A2A Task:")
    print(task.to_json())

    # Simulate sending a task (requires 'requests' to be installed for real calls)
    # a2a_client = A2AClient()
    # import asyncio
    # try:
    #     response = asyncio.run(a2a_client.send_task_request(agent_card, task))
    #     print("\nSimulated Task Request Response:")
    #     print(json.dumps(response, indent=2))
    # except ConnectionError as ce:
    #     print(f"\nError during simulated task request: {ce}")
    # except ImportError:
    #     print("\nSkipping A2AClient send_task_request simulation as 'requests' is not installed.")

import pytest
from src.server import system_prompt_for_agent, BedrockLogsParams, EC2Params  # Import necessary items

# Mock boto3 client if needed for other tests in the future
# class MockBoto3Client:
#     def get_caller_identity(self):
#         return {'Account': '123456789012'}

# @pytest.fixture
# def mock_boto3(monkeypatch):
#     monkeypatch.setattr(boto3, 'client', lambda service: MockBoto3Client() if service == 'sts' else None)


def test_system_prompt_for_agent_no_account():
    # Assuming the function falls back to boto3 if no account is provided
    # This test might fail if AWS credentials are not configured where the test runs
    # or if we don't mock boto3. For now, let's test the explicit account case.
    pass # Skipping this version for now as it requires mocking or credentials

def test_system_prompt_for_agent_with_account():
    account_id = "987654321098"
    prompt = system_prompt_for_agent(aws_account_id=account_id)
    assert f"for account {account_id}" in prompt
    assert "You are an expert AWS cost analyst AI agent" in prompt

# Add more tests here for other functions as needed
# Example placeholder for testing Pydantic models
def test_bedrock_logs_params_defaults():
    params = BedrockLogsParams()
    assert params.days == 7
    assert params.region == "us-east-1"
    assert params.log_group_name == "BedrockModelInvocationLogGroup" # Default might vary based on env var
    assert params.aws_account_id is None

def test_ec2_params_defaults():
    params = EC2Params()
    assert params.days == 1
    assert params.region == "us-east-1"
    assert params.aws_account_id is None

# Consider adding tests for get_aws_service_boto3_client with mocking
# Consider adding tests for get_bedrock_logs with mocking (using moto or similar)
# Consider adding tests for the MCP tools, potentially mocking the underlying functions 
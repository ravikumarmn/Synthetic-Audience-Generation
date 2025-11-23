# Test Suite for Synthetic Audience MVP

This directory contains comprehensive test cases for the `synthetic_audience_mvp.py` module.

## Test Structure

### Test Files

- **`conftest.py`** - Test configuration, fixtures, and shared test data
- **`test_models.py`** - Tests for Pydantic models (FilterDetails, InputPersona, etc.)
- **`test_generators.py`** - Tests for LLM content generators (sync and async)
- **`test_processors.py`** - Tests for DistributionCalculator and PersonaProcessor
- **`test_workflow_nodes.py`** - Tests for individual workflow nodes
- **`test_main_generator.py`** - Tests for SyntheticAudienceGenerator main class
- **`test_cli.py`** - Tests for CLI functionality and argument parsing

### Test Categories

#### Unit Tests
- **Model Validation**: Pydantic model validation and serialization
- **Content Generation**: LLM content generation with mocked responses
- **Distribution Logic**: Demographic distribution calculations
- **Template Extraction**: Persona processing and template creation
- **Workflow Nodes**: Individual node functionality

#### Integration Tests
- **End-to-End Workflows**: Complete workflow execution
- **File I/O Operations**: JSON loading and writing
- **Async Processing**: Parallel batch processing with real async operations

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestFilterDetails

# Run specific test method
pytest tests/test_models.py::TestFilterDetails::test_valid_filter_details
```

### Using the Test Runner

```bash
# Run all tests with the custom runner
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file test_models

# Run in parallel
python run_tests.py --parallel 4

# Run only failed tests from last run
python run_tests.py --lf
```

### Test Selection

```bash
# Run fast tests only (exclude slow tests)
pytest -m "not slow"

# Run only async tests
pytest -m asyncio

# Run integration tests only
pytest -m integration
```

## Test Coverage

The test suite aims for comprehensive coverage of:

### ✅ Covered Components

1. **Pydantic Models** (100% coverage)
   - FilterDetails validation and proportions
   - InputPersona and RequestData parsing
   - BehavioralContent and SyntheticProfile creation
   - ProcessingTemplates and DemographicAssignment

2. **Content Generators** (95% coverage)
   - LLMContentGenerator with Google/Azure providers
   - AsyncLLMContentGenerator with ThreadPoolExecutor
   - ParallelBatchProcessor with semaphore control
   - Error handling and exception management

3. **Processors and Calculators** (100% coverage)
   - DistributionCalculator demographic scheduling
   - PersonaProcessor template extraction
   - Content filtering and deduplication

4. **Workflow Nodes** (90% coverage)
   - JSON loading and validation
   - Distribution building and persona processing
   - Profile generation (sync and async)
   - Output writing and finalization

5. **Main Generator Class** (95% coverage)
   - SyntheticAudienceGenerator initialization
   - Workflow creation and configuration
   - Request processing and statistics

6. **CLI Functionality** (85% coverage)
   - Argument parsing and validation
   - Configuration options and flags
   - Help messages and examples

### Test Fixtures and Mocks

The test suite uses comprehensive fixtures for:

- **Sample Data**: Realistic test data for all model types
- **Temporary Files**: Safe file I/O testing with cleanup
- **Mock LLMs**: Simulated LLM responses for testing without API calls
- **Environment Setup**: Consistent test environment configuration

## Test Configuration

### Environment Variables

Tests use isolated environment variables:

```bash
GOOGLE_API_KEY=test_google_key
AZURE_OPENAI_ENDPOINT=https://test.openai.azure.com/
AZURE_OPENAI_API_KEY=test_azure_key
AZURE_OPENAI_DEPLOYMENT_NAME=test-deployment
LLM_PROVIDER=google
PARALLEL_BATCH_SIZE=3
MAX_WORKERS=2
CONCURRENT_REQUESTS=2
```

### Pytest Configuration

See `pytest.ini` for:
- Test discovery patterns
- Marker definitions
- Warning filters
- Async test configuration

## Writing New Tests

### Test Naming Convention

```python
class TestClassName:
    def test_method_name_scenario(self):
        """Test description."""
        pass
```

### Using Fixtures

```python
def test_with_fixtures(self, sample_filter_details, temp_input_file):
    """Test using predefined fixtures."""
    # Use fixtures directly
    assert sample_filter_details.user_req_responses > 0
```

### Mocking External Dependencies

```python
@patch('synthetic_audience_mvp.ChatGoogleGenerativeAI')
def test_with_mock(self, mock_llm):
    """Test with mocked LLM."""
    mock_llm.return_value.invoke.return_value.content = '{"test": "data"}'
    # Test code here
```

### Async Test Methods

```python
@pytest.mark.asyncio
async def test_async_functionality(self):
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python run_tests.py --coverage --parallel 4
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` directory is in Python path
2. **Async Test Failures**: Check pytest-asyncio installation
3. **Mock Issues**: Verify patch targets match actual import paths
4. **File Permission Errors**: Ensure test has write permissions for temp files

### Debug Mode

```bash
# Run with debugger on failure
pytest --pdb

# Run with detailed output
pytest -vv --tb=long

# Run single test with debugging
pytest tests/test_models.py::TestFilterDetails::test_valid_filter_details -vv --pdb
```

## Performance Testing

For performance testing of the actual LLM integration:

```bash
# Run performance tests (requires API keys)
pytest -m "slow" --durations=10

# Profile test execution
pytest --profile --profile-svg
```

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure >90% code coverage for new code
3. Add appropriate test markers
4. Update this README if adding new test categories
5. Run full test suite before committing

```bash
# Pre-commit test run
python run_tests.py --coverage --fast
```

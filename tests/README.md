# Customer Churn Prediction - Test Suite

This directory contains comprehensive tests for the Customer Churn Prediction system.

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_domain_services.py      # Domain layer tests
│   └── test_application_use_cases.py # Application layer tests
├── integration/             # Integration tests
│   └── test_api_endpoints.py        # API endpoint tests
├── fixtures/               # Test data and fixtures
└── conftest.py            # Pytest configuration and fixtures
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# API tests only
pytest tests/integration/test_api_endpoints.py
```

### Run with coverage
```bash
pytest --cov=src tests/
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test markers
```bash
# Unit tests
pytest -m unit

# Integration tests
pytest -m integration

# Slow tests
pytest -m slow
```

## Test Categories

### Unit Tests

#### Domain Layer Tests (`test_domain_services.py`)
- **Customer Entity**: Creation, validation, data conversion
- **ChurnAnalysisService**: Risk scoring, customer segmentation, recommendations
- **CsvCustomerRepository**: Data loading, saving, CRUD operations
- **SklearnChurnPredictionModel**: Model loading, predictions, confidence calculation

#### Application Layer Tests (`test_application_use_cases.py`)
- **PredictCustomerChurnUseCase**: Single prediction logic, error handling
- **TrainChurnModelUseCase**: Model training pipeline, validation
- **ProcessCustomerBatchUseCase**: Batch processing, performance, error handling

### Integration Tests

#### API Endpoints (`test_api_endpoints.py`)
- **Health Checks**: Basic and detailed health endpoints
- **Model Information**: Model metadata and performance metrics
- **Single Prediction**: Individual customer prediction endpoint
- **Batch Prediction**: Multiple customer prediction endpoint
- **CSV Upload**: File-based prediction endpoint
- **Error Handling**: Various error scenarios and responses
- **CORS**: Cross-origin resource sharing configuration

## Test Data

### Sample Customer Data
Tests use synthetic customer data that mimics real-world scenarios:

```json
{
    "customer_id": "CUST-001",
    "gender": "Female",
    "senior_citizen": false,
    "partner": true,
    "dependents": false,
    "phone_service": true,
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract_type": "Month-to-month",
    "paperless_billing": true,
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 1510.45,
    "tenure_months": 24
}
```

### Test Scenarios

#### High-Risk Customers
- Month-to-month contracts
- High monthly charges
- Short tenure
- Limited additional services

#### Low-Risk Customers
- Long-term contracts
- Lower monthly charges
- Longer tenure
- Multiple additional services

#### Edge Cases
- Missing data fields
- Invalid data types
- Extreme values
- Empty batches
- Large datasets (100+ customers)

## Mocking Strategy

Tests use extensive mocking to isolate components:

### Domain Layer Mocks
- Repository interfaces
- Model interfaces
- Service dependencies

### Application Layer Mocks
- Use case dependencies
- External service calls
- Data access operations

### Integration Layer Mocks
- Database connections
- File system operations
- External API calls

## Performance Testing

### Load Testing
- Large batch processing (100+ customers)
- Concurrent request handling
- Memory usage monitoring

### Response Time Validation
- API endpoint response times
- Model prediction latency
- Database query performance

## Error Handling Tests

### Validation Errors
- Invalid data formats
- Missing required fields
- Type mismatches
- Value range violations

### System Errors
- Model loading failures
- Database connection issues
- File system errors
- Network timeouts

### Business Logic Errors
- Insufficient training data
- Model accuracy below threshold
- Invalid prediction requests

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Output formatting
- Warning filters
- Marker definitions

### Coverage Reporting
- Source code coverage
- Branch coverage
- Line coverage
- Missing coverage analysis

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

### GitHub Actions
- Automated test execution
- Coverage reporting
- Performance benchmarks
- Security scanning

### Pre-commit Hooks
- Test execution before commits
- Code quality checks
- Documentation validation

## Best Practices

### Test Isolation
- Each test is independent
- No shared state between tests
- Proper cleanup after tests

### Test Data Management
- Consistent test data
- Reproducible results
- Data privacy considerations

### Naming Conventions
- Descriptive test names
- Clear assertion messages
- Organized test structure

### Documentation
- Test purpose documentation
- Expected behavior descriptions
- Failure scenario explanations

## Troubleshooting

### Common Issues
1. **Model Loading**: Ensure model files are available
2. **Database Connection**: Check database configuration
3. **File Permissions**: Verify file access rights
4. **Memory Usage**: Monitor memory consumption

### Debug Mode
```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run specific test with debugging
pytest tests/unit/test_domain_services.py::TestCustomer::test_customer_creation -v -s
```

### Performance Issues
```bash
# Profile test execution
pytest --durations=0

# Run slow tests only
pytest -m slow
```

## Future Enhancements

### Additional Test Categories
- **Load Testing**: High-volume scenario testing
- **Security Testing**: Vulnerability assessment
- **Compatibility Testing**: Cross-platform validation
- **Regression Testing**: Historical bug validation

### Test Automation
- **Visual Testing**: UI component validation
- **Contract Testing**: API contract validation
- **Chaos Testing**: System resilience testing
- **Mutation Testing**: Test quality assessment
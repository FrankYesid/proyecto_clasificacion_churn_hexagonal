import pytest
import json
import pandas as pd
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from src.interface.api.main import app
from src.application.use_cases.predict_customer_churn import PredictCustomerChurnUseCase
from src.application.use_cases.process_customer_batch import ProcessCustomerBatchUseCase
from src.application.dto import CustomerInputDTO, BatchPredictionInputDTO, CustomerOutputDTO, BatchPredictionOutputDTO


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        "customer_id": "CUST-001",
        "gender": "Female",
        "senior_citizen": False,
        "partner": True,
        "dependents": False,
        "phone_service": True,
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract_type": "Month-to-month",
        "paperless_billing": True,
        "payment_method": "Electronic check",
        "monthly_charges": 70.35,
        "total_charges": 1510.45,
        "tenure_months": 24
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return {
        "batch_id": "BATCH-001",
        "customers": [
            {
                "customer_id": "CUST-001",
                "gender": "Female",
                "senior_citizen": False,
                "partner": True,
                "dependents": False,
                "phone_service": True,
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract_type": "Month-to-month",
                "paperless_billing": True,
                "payment_method": "Electronic check",
                "monthly_charges": 70.35,
                "total_charges": 1510.45,
                "tenure_months": 24
            },
            {
                "customer_id": "CUST-002",
                "gender": "Male",
                "senior_citizen": True,
                "partner": False,
                "dependents": True,
                "phone_service": False,
                "multiple_lines": "No",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract_type": "Two year",
                "paperless_billing": False,
                "payment_method": "Bank transfer (automatic)",
                "monthly_charges": 45.99,
                "total_charges": 1200.50,
                "tenure_months": 48
            }
        ]
    }


class TestHealthEndpoints:
    """Test cases for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
        assert data["service"] == "churn-prediction-api"
    
    def test_health_detailed(self, client):
        """Test detailed health check."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "model_loaded" in data
        assert "memory_usage_mb" in data


class TestModelEndpoints:
    """Test cases for model information endpoints."""
    
    def test_model_info(self, client):
        """Test model information endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert "training_date" in data
        assert "features" in data
        assert "performance_metrics" in data
        assert "last_updated" in data
    
    def test_model_info_not_loaded(self, client):
        """Test model info when model is not loaded."""
        with patch('src.interface.api.main.model') as mock_model:
            mock_model.is_loaded.return_value = False
            
            response = client.get("/model/info")
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]


class TestPredictionEndpoints:
    """Test cases for prediction endpoints."""
    
    @patch('src.interface.api.main.predict_use_case')
    def test_predict_single_success(self, mock_predict_use_case, client, sample_customer_data):
        """Test successful single prediction."""
        # Mock use case response
        mock_response = CustomerOutputDTO(
            customer_id="CUST-001",
            churn_probability=0.85,
            confidence=0.92,
            risk_level="High",
            customer_segment="Premium",
            recommendations=["Offer loyalty discount", "Provide premium support"]
        )
        mock_predict_use_case.execute.return_value = mock_response
        
        response = client.post("/predict", json=sample_customer_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "CUST-001"
        assert data["churn_probability"] == 0.85
        assert data["confidence"] == 0.92
        assert data["risk_level"] == "High"
        assert data["customer_segment"] == "Premium"
        assert len(data["recommendations"]) == 2
        
        # Verify use case was called
        mock_predict_use_case.execute.assert_called_once()
    
    @patch('src.interface.api.main.predict_use_case')
    def test_predict_single_validation_error(self, mock_predict_use_case, client, sample_customer_data):
        """Test prediction with validation error."""
        # Modify data to be invalid
        invalid_data = sample_customer_data.copy()
        invalid_data["monthly_charges"] = -10.0  # Invalid negative value
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @patch('src.interface.api.main.predict_use_case')
    def test_predict_single_model_error(self, mock_predict_use_case, client, sample_customer_data):
        """Test prediction when model fails."""
        # Mock use case to raise exception
        mock_predict_use_case.execute.side_effect = RuntimeError("Model prediction failed")
        
        response = client.post("/predict", json=sample_customer_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Model prediction failed" in data["detail"]
    
    @patch('src.interface.api.main.batch_use_case')
    def test_predict_batch_success(self, mock_batch_use_case, client, sample_batch_data):
        """Test successful batch prediction."""
        # Mock use case response
        mock_response = BatchPredictionOutputDTO(
            batch_id="BATCH-001",
            total_customers=2,
            processed_at="2024-01-01T12:00:00Z",
            processing_time_seconds=0.5,
            predictions=[
                CustomerOutputDTO(
                    customer_id="CUST-001",
                    churn_probability=0.85,
                    confidence=0.92,
                    risk_level="High",
                    customer_segment="Premium",
                    recommendations=["Offer loyalty discount"]
                ),
                CustomerOutputDTO(
                    customer_id="CUST-002",
                    churn_probability=0.20,
                    confidence=0.88,
                    risk_level="Low",
                    customer_segment="Basic",
                    recommendations=["Upsell additional services"]
                )
            ]
        )
        mock_batch_use_case.execute.return_value = mock_response
        
        response = client.post("/predict/batch", json=sample_batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == "BATCH-001"
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2
        assert data["processing_time_seconds"] == 0.5
        
        # Verify predictions
        assert data["predictions"][0]["customer_id"] == "CUST-001"
        assert data["predictions"][0]["churn_probability"] == 0.85
        assert data["predictions"][0]["risk_level"] == "High"
        
        assert data["predictions"][1]["customer_id"] == "CUST-002"
        assert data["predictions"][1]["churn_probability"] == 0.20
        assert data["predictions"][1]["risk_level"] == "Low"
    
    @patch('src.interface.api.main.batch_use_case')
    def test_predict_batch_empty(self, mock_batch_use_case, client):
        """Test batch prediction with empty batch."""
        empty_batch = {
            "batch_id": "BATCH-EMPTY",
            "customers": []
        }
        
        # Mock use case response
        mock_response = BatchPredictionOutputDTO(
            batch_id="BATCH-EMPTY",
            total_customers=0,
            processed_at="2024-01-01T12:00:00Z",
            processing_time_seconds=0.1,
            predictions=[]
        )
        mock_batch_use_case.execute.return_value = mock_response
        
        response = client.post("/predict/batch", json=empty_batch)
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == "BATCH-EMPTY"
        assert data["total_customers"] == 0
        assert len(data["predictions"]) == 0
    
    @patch('src.interface.api.main.batch_use_case')
    def test_predict_batch_large(self, mock_batch_use_case, client):
        """Test batch prediction with large batch."""
        # Create large batch
        large_batch = {
            "batch_id": "BATCH-LARGE",
            "customers": [
                {
                    "customer_id": f"CUST-{i:03d}",
                    "gender": "Female" if i % 2 == 0 else "Male",
                    "senior_citizen": i % 3 == 0,
                    "partner": i % 2 == 0,
                    "dependents": i % 3 == 0,
                    "phone_service": True,
                    "multiple_lines": "No",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "online_backup": "Yes",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "Yes",
                    "streaming_movies": "No",
                    "contract_type": "Month-to-month",
                    "paperless_billing": True,
                    "payment_method": "Electronic check",
                    "monthly_charges": 70.35 + i * 5,
                    "total_charges": 1510.45 + i * 100,
                    "tenure_months": 24 + i * 2
                } for i in range(100)
            ]
        }
        
        # Mock use case response
        mock_predictions = [
            CustomerOutputDTO(
                customer_id=f"CUST-{i:03d}",
                churn_probability=0.50 + (i % 50) * 0.01,
                confidence=0.90,
                risk_level="High" if i % 3 == 0 else "Medium" if i % 3 == 1 else "Low",
                customer_segment="Standard",
                recommendations=["General recommendation"]
            ) for i in range(100)
        ]
        
        mock_response = BatchPredictionOutputDTO(
            batch_id="BATCH-LARGE",
            total_customers=100,
            processed_at="2024-01-01T12:00:00Z",
            processing_time_seconds=2.5,
            predictions=mock_predictions
        )
        mock_batch_use_case.execute.return_value = mock_response
        
        response = client.post("/predict/batch", json=large_batch)
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == "BATCH-LARGE"
        assert data["total_customers"] == 100
        assert len(data["predictions"]) == 100
        assert data["processing_time_seconds"] == 2.5
    
    def test_predict_csv_success(self, client):
        """Test CSV file prediction."""
        # Create sample CSV content
        csv_content = """customer_id,gender,senior_citizen,partner,dependents,phone_service,multiple_lines,internet_service,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract_type,paperless_billing,payment_method,monthly_charges,total_charges,tenure_months
CUST-001,Female,False,True,False,True,No,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,True,Electronic check,70.35,1510.45,24
CUST-002,Male,True,False,True,False,No,DSL,Yes,No,Yes,Yes,No,No,Two year,False,Bank transfer (automatic),45.99,1200.50,48"""
        
        # Mock batch use case
        with patch('src.interface.api.main.batch_use_case') as mock_batch_use_case:
            mock_response = BatchPredictionOutputDTO(
                batch_id="CSV-BATCH-001",
                total_customers=2,
                processed_at="2024-01-01T12:00:00Z",
                processing_time_seconds=1.0,
                predictions=[
                    CustomerOutputDTO(
                        customer_id="CUST-001",
                        churn_probability=0.85,
                        confidence=0.92,
                        risk_level="High",
                        customer_segment="Premium",
                        recommendations=["Offer loyalty discount"]
                    ),
                    CustomerOutputDTO(
                        customer_id="CUST-002",
                        churn_probability=0.20,
                        confidence=0.88,
                        risk_level="Low",
                        customer_segment="Basic",
                        recommendations=["Upsell additional services"]
                    )
                ]
            )
            mock_batch_use_case.execute.return_value = mock_response
            
            # Create file-like object
            from io import BytesIO
            csv_file = BytesIO(csv_content.encode('utf-8'))
            
            response = client.post(
                "/predict/csv",
                files={"file": ("test_customers.csv", csv_file, "text/csv")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["batch_id"] == "CSV-BATCH-001"
            assert data["total_customers"] == 2
            assert len(data["predictions"]) == 2
    
    def test_predict_csv_no_file(self, client):
        """Test CSV prediction without file."""
        response = client.post("/predict/csv")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_predict_csv_invalid_format(self, client):
        """Test CSV prediction with invalid file format."""
        # Create invalid file content
        invalid_content = "This is not a CSV file"
        
        from io import BytesIO
        invalid_file = BytesIO(invalid_content.encode('utf-8'))
        
        response = client.post(
            "/predict/csv",
            files={"file": ("invalid.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "CSV format" in data["detail"]


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_404_endpoint(self, client):
        """Test non-existent endpoint."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.put("/health")
        
        assert response.status_code == 405
        data = response.json()
        assert "detail" in data
    
    def test_invalid_json(self, client):
        """Test invalid JSON payload."""
        response = client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestCORS:
    """Test cases for CORS configuration."""
    
    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_cors_actual_request(self, client, sample_customer_data):
        """Test CORS for actual request."""
        with patch('src.interface.api.main.predict_use_case') as mock_predict_use_case:
            mock_response = CustomerOutputDTO(
                customer_id="CUST-001",
                churn_probability=0.85,
                confidence=0.92,
                risk_level="High",
                customer_segment="Premium",
                recommendations=["Offer loyalty discount"]
            )
            mock_predict_use_case.execute.return_value = mock_response
            
            response = client.post(
                "/predict",
                json=sample_customer_data,
                headers={"Origin": "http://localhost:3000"}
            )
            
            assert response.status_code == 200
            assert "access-control-allow-origin" in response.headers
            assert response.headers["access-control-allow-origin"] == "*"
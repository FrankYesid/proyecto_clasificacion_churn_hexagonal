import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.application.use_cases.predict_customer_churn import PredictCustomerChurnUseCase
from src.application.use_cases.train_churn_model import TrainChurnModelUseCase
from src.application.use_cases.process_customer_batch import ProcessCustomerBatchUseCase
from src.application.dto import CustomerInputDTO, BatchPredictionInputDTO
from src.domain.entities.customer import Customer
from src.domain.repositories.customer_repository import CustomerRepository
from src.domain.models.churn_prediction_model import ChurnPredictionModel
from src.domain.services.churn_analysis_service import ChurnAnalysisService


class TestPredictCustomerChurnUseCase:
    """Test cases for PredictCustomerChurnUseCase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ChurnPredictionModel)
        self.mock_analysis_service = Mock(spec=ChurnAnalysisService)
        self.use_case = PredictCustomerChurnUseCase(
            model=self.mock_model,
            analysis_service=self.mock_analysis_service
        )
        
        self.sample_dto = CustomerInputDTO(
            customer_id='CUST-001',
            gender='Female',
            senior_citizen=False,
            partner=True,
            dependents=False,
            phone_service=True,
            multiple_lines='No',
            internet_service='Fiber optic',
            online_security='No',
            online_backup='Yes',
            device_protection='No',
            tech_support='No',
            streaming_tv='Yes',
            streaming_movies='No',
            contract_type='Month-to-month',
            paperless_billing=True,
            payment_method='Electronic check',
            monthly_charges=70.35,
            total_charges=1510.45,
            tenure_months=24
        )
    
    def test_execute_success(self):
        """Test successful prediction execution."""
        # Mock model prediction
        self.mock_model.predict_single.return_value = {
            'churn_probability': 0.85,
            'confidence': 0.92,
            'risk_level': 'High'
        }
        
        # Mock analysis service
        self.mock_analysis_service.get_customer_segment.return_value = 'Premium'
        self.mock_analysis_service.generate_recommendations.return_value = [
            'Offer loyalty discount',
            'Provide premium customer support'
        ]
        
        result = self.use_case.execute(self.sample_dto)
        
        assert result.customer_id == 'CUST-001'
        assert result.churn_probability == 0.85
        assert result.confidence == 0.92
        assert result.risk_level == 'High'
        assert result.customer_segment == 'Premium'
        assert len(result.recommendations) == 2
        assert 'Offer loyalty discount' in result.recommendations
        
        # Verify model was called
        self.mock_model.predict_single.assert_called_once()
        self.mock_analysis_service.get_customer_segment.assert_called_once()
        self.mock_analysis_service.generate_recommendations.assert_called_once()
    
    def test_execute_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        self.mock_model.predict_single.side_effect = RuntimeError("Model not loaded")
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.use_case.execute(self.sample_dto)
    
    def test_execute_invalid_input(self):
        """Test prediction with invalid input."""
        invalid_dto = CustomerInputDTO(
            customer_id='',
            gender='Invalid',
            senior_citizen=False,
            partner=True,
            dependents=False,
            phone_service=True,
            multiple_lines='No',
            internet_service='Fiber optic',
            online_security='No',
            online_backup='Yes',
            device_protection='No',
            tech_support='No',
            streaming_tv='Yes',
            streaming_movies='No',
            contract_type='Month-to-month',
            paperless_billing=True,
            payment_method='Electronic check',
            monthly_charges=-10.0,  # Invalid negative value
            total_charges=1510.45,
            tenure_months=24
        )
        
        with pytest.raises(ValueError):
            self.use_case.execute(invalid_dto)


class TestTrainChurnModelUseCase:
    """Test cases for TrainChurnModelUseCase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ChurnPredictionModel)
        self.mock_repository = Mock(spec=CustomerRepository)
        self.mock_analysis_service = Mock(spec=ChurnAnalysisService)
        self.use_case = TrainChurnModelUseCase(
            model=self.mock_model,
            repository=self.mock_repository,
            analysis_service=self.mock_analysis_service
        )
        
        # Sample training data
        self.sample_customers = [
            Customer(
                customer_id=f'CUST-{i:03d}',
                gender='Female' if i % 2 == 0 else 'Male',
                senior_citizen=i % 3 == 0,
                partner=i % 2 == 0,
                dependents=i % 3 == 0,
                phone_service=True,
                multiple_lines='No',
                internet_service='Fiber optic',
                online_security='No',
                online_backup='Yes',
                device_protection='No',
                tech_support='No',
                streaming_tv='Yes',
                streaming_movies='No',
                contract_type='Month-to-month',
                paperless_billing=True,
                payment_method='Electronic check',
                monthly_charges=70.35 + i * 5,
                total_charges=1510.45 + i * 100,
                tenure_months=24 + i * 2
            ) for i in range(10)
        ]
    
    def test_execute_success(self):
        """Test successful model training."""
        # Mock repository
        self.mock_repository.load_customers.return_value = self.sample_customers
        
        # Mock model training
        self.mock_model.train.return_value = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.87
        }
        
        result = self.use_case.execute()
        
        assert result['accuracy'] == 0.85
        assert result['precision'] == 0.80
        assert result['recall'] == 0.75
        assert result['f1_score'] == 0.77
        assert result['roc_auc'] == 0.87
        
        # Verify dependencies were called
        self.mock_repository.load_customers.assert_called_once()
        self.mock_model.train.assert_called_once()
    
    def test_execute_insufficient_data(self):
        """Test training with insufficient data."""
        # Mock repository with very few customers
        self.mock_repository.load_customers.return_value = self.sample_customers[:2]
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            self.use_case.execute()
    
    def test_execute_model_training_failure(self):
        """Test model training failure."""
        # Mock repository
        self.mock_repository.load_customers.return_value = self.sample_customers
        
        # Mock model training failure
        self.mock_model.train.side_effect = RuntimeError("Training failed")
        
        with pytest.raises(RuntimeError, match="Training failed"):
            self.use_case.execute()
    
    def test_execute_with_validation_split(self):
        """Test training with validation split."""
        # Mock repository
        self.mock_repository.load_customers.return_value = self.sample_customers
        
        # Mock model training
        self.mock_model.train.return_value = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.87
        }
        
        result = self.use_case.execute(validation_split=0.2)
        
        assert result['accuracy'] == 0.85
        # Verify model was called with validation split
        self.mock_model.train.assert_called_once()
        call_args = self.mock_model.train.call_args
        assert 'validation_split' in call_args.kwargs
        assert call_args.kwargs['validation_split'] == 0.2


class TestProcessCustomerBatchUseCase:
    """Test cases for ProcessCustomerBatchUseCase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ChurnPredictionModel)
        self.mock_analysis_service = Mock(spec=ChurnAnalysisService)
        self.use_case = ProcessCustomerBatchUseCase(
            model=self.mock_model,
            analysis_service=self.mock_analysis_service
        )
        
        # Sample batch input
        self.sample_batch = BatchPredictionInputDTO(
            batch_id='BATCH-001',
            customers=[
                CustomerInputDTO(
                    customer_id='CUST-001',
                    gender='Female',
                    senior_citizen=False,
                    partner=True,
                    dependents=False,
                    phone_service=True,
                    multiple_lines='No',
                    internet_service='Fiber optic',
                    online_security='No',
                    online_backup='Yes',
                    device_protection='No',
                    tech_support='No',
                    streaming_tv='Yes',
                    streaming_movies='No',
                    contract_type='Month-to-month',
                    paperless_billing=True,
                    payment_method='Electronic check',
                    monthly_charges=70.35,
                    total_charges=1510.45,
                    tenure_months=24
                ),
                CustomerInputDTO(
                    customer_id='CUST-002',
                    gender='Male',
                    senior_citizen=True,
                    partner=False,
                    dependents=True,
                    phone_service=False,
                    multiple_lines='No',
                    internet_service='DSL',
                    online_security='Yes',
                    online_backup='No',
                    device_protection='Yes',
                    tech_support='Yes',
                    streaming_tv='No',
                    streaming_movies='No',
                    contract_type='Two year',
                    paperless_billing=False,
                    payment_method='Bank transfer (automatic)',
                    monthly_charges=45.99,
                    total_charges=1200.50,
                    tenure_months=48
                )
            ]
        )
    
    def test_execute_success(self):
        """Test successful batch processing."""
        # Mock model predictions
        self.mock_model.predict_batch.return_value = [
            {
                'churn_probability': 0.85,
                'confidence': 0.92,
                'risk_level': 'High'
            },
            {
                'churn_probability': 0.20,
                'confidence': 0.88,
                'risk_level': 'Low'
            }
        ]
        
        # Mock analysis service
        self.mock_analysis_service.get_customer_segment.side_effect = ['Premium', 'Basic']
        self.mock_analysis_service.generate_recommendations.side_effect = [
            ['Offer loyalty discount', 'Provide premium customer support'],
            ['Upsell additional services', 'Offer family plans']
        ]
        
        result = self.use_case.execute(self.sample_batch)
        
        assert result.batch_id == 'BATCH-001'
        assert result.total_customers == 2
        assert result.processed_at is not None
        assert len(result.predictions) == 2
        
        # Check first prediction
        assert result.predictions[0].customer_id == 'CUST-001'
        assert result.predictions[0].churn_probability == 0.85
        assert result.predictions[0].risk_level == 'High'
        assert result.predictions[0].customer_segment == 'Premium'
        
        # Check second prediction
        assert result.predictions[1].customer_id == 'CUST-002'
        assert result.predictions[1].churn_probability == 0.20
        assert result.predictions[1].risk_level == 'Low'
        assert result.predictions[1].customer_segment == 'Basic'
        
        # Verify dependencies were called
        self.mock_model.predict_batch.assert_called_once()
        assert self.mock_analysis_service.get_customer_segment.call_count == 2
        assert self.mock_analysis_service.generate_recommendations.call_count == 2
    
    def test_execute_empty_batch(self):
        """Test processing empty batch."""
        empty_batch = BatchPredictionInputDTO(
            batch_id='BATCH-EMPTY',
            customers=[]
        )
        
        result = self.use_case.execute(empty_batch)
        
        assert result.batch_id == 'BATCH-EMPTY'
        assert result.total_customers == 0
        assert len(result.predictions) == 0
    
    def test_execute_large_batch(self):
        """Test processing large batch."""
        # Create large batch
        large_batch = BatchPredictionInputDTO(
            batch_id='BATCH-LARGE',
            customers=[
                CustomerInputDTO(
                    customer_id=f'CUST-{i:03d}',
                    gender='Female' if i % 2 == 0 else 'Male',
                    senior_citizen=i % 3 == 0,
                    partner=i % 2 == 0,
                    dependents=i % 3 == 0,
                    phone_service=True,
                    multiple_lines='No',
                    internet_service='Fiber optic',
                    online_security='No',
                    online_backup='Yes',
                    device_protection='No',
                    tech_support='No',
                    streaming_tv='Yes',
                    streaming_movies='No',
                    contract_type='Month-to-month',
                    paperless_billing=True,
                    payment_method='Electronic check',
                    monthly_charges=70.35 + i * 5,
                    total_charges=1510.45 + i * 100,
                    tenure_months=24 + i * 2
                ) for i in range(100)
            ]
        )
        
        # Mock model predictions for large batch
        self.mock_model.predict_batch.return_value = [
            {
                'churn_probability': 0.50 + (i % 50) * 0.01,
                'confidence': 0.90,
                'risk_level': 'High' if i % 3 == 0 else 'Medium' if i % 3 == 1 else 'Low'
            } for i in range(100)
        ]
        
        # Mock analysis service
        self.mock_analysis_service.get_customer_segment.return_value = 'Standard'
        self.mock_analysis_service.generate_recommendations.return_value = ['General recommendation']
        
        result = self.use_case.execute(large_batch)
        
        assert result.batch_id == 'BATCH-LARGE'
        assert result.total_customers == 100
        assert len(result.predictions) == 100
        assert result.processing_time_seconds > 0
    
    def test_execute_with_model_failure(self):
        """Test batch processing with model failure."""
        # Mock model failure
        self.mock_model.predict_batch.side_effect = RuntimeError("Model prediction failed")
        
        with pytest.raises(RuntimeError, match="Model prediction failed"):
            self.use_case.execute(self.sample_batch)
    
    def test_execute_with_partial_failure(self):
        """Test batch processing with partial failure."""
        # Mock model to return None for some predictions (simulating partial failure)
        self.mock_model.predict_batch.return_value = [
            {
                'churn_probability': 0.85,
                'confidence': 0.92,
                'risk_level': 'High'
            },
            None  # Simulate failure for second customer
        ]
        
        # Mock analysis service for successful prediction
        self.mock_analysis_service.get_customer_segment.return_value = 'Premium'
        self.mock_analysis_service.generate_recommendations.return_value = ['Offer loyalty discount']
        
        result = self.use_case.execute(self.sample_batch)
        
        assert result.batch_id == 'BATCH-001'
        assert result.total_customers == 2
        assert len(result.predictions) == 1  # Only one successful prediction
        assert result.predictions[0].customer_id == 'CUST-001'
        assert result.failed_customers == 1
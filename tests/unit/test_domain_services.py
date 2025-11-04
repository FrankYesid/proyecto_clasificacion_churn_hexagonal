import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.domain.entities.customer import Customer
from src.domain.services.churn_analysis_service import ChurnAnalysisService
from src.adapters.repositories.csv_customer_repository import CsvCustomerRepository
from src.adapters.machine_learning.sklearn_churn_model import SklearnChurnPredictionModel


class TestCustomer:
    """Test cases for Customer entity."""
    
    def test_customer_creation(self):
        """Test customer entity creation with valid data."""
        customer_data = {
            'customer_id': 'CUST-001',
            'gender': 'Female',
            'senior_citizen': False,
            'partner': True,
            'dependents': False,
            'phone_service': True,
            'multiple_lines': 'No',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'online_backup': 'Yes',
            'device_protection': 'No',
            'tech_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'No',
            'contract_type': 'Month-to-month',
            'paperless_billing': True,
            'payment_method': 'Electronic check',
            'monthly_charges': 70.35,
            'total_charges': 1510.45,
            'tenure_months': 24
        }
        
        customer = Customer(**customer_data)
        
        assert customer.customer_id == 'CUST-001'
        assert customer.gender == 'Female'
        assert customer.monthly_charges == 70.35
        assert customer.tenure_months == 24
    
    def test_customer_to_dict(self):
        """Test customer conversion to dictionary."""
        customer = Customer(
            customer_id='CUST-001',
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
            tenure_months=36
        )
        
        customer_dict = customer.to_dict()
        
        assert customer_dict['customer_id'] == 'CUST-001'
        assert customer_dict['gender'] == 'Male'
        assert customer_dict['monthly_charges'] == 45.99
        assert customer_dict['tenure_months'] == 36
    
    def test_customer_validation(self):
        """Test customer data validation."""
        # Test invalid monthly charges
        with pytest.raises(ValueError):
            Customer(
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
                monthly_charges=-10.0,  # Invalid negative value
                total_charges=1510.45,
                tenure_months=24
            )
        
        # Test invalid tenure
        with pytest.raises(ValueError):
            Customer(
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
                tenure_months=-5  # Invalid negative value
            )


class TestChurnAnalysisService:
    """Test cases for ChurnAnalysisService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = ChurnAnalysisService()
        self.sample_customer = Customer(
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
    
    def test_calculate_risk_score_high_risk(self):
        """Test risk score calculation for high-risk customer."""
        risk_score = self.service.calculate_risk_score(self.sample_customer)
        
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0
        # High-risk indicators: month-to-month contract, high monthly charges
        assert risk_score > 0.5
    
    def test_calculate_risk_score_low_risk(self):
        """Test risk score calculation for low-risk customer."""
        low_risk_customer = Customer(
            customer_id='CUST-002',
            gender='Male',
            senior_citizen=False,
            partner=True,
            dependents=True,
            phone_service=True,
            multiple_lines='Yes',
            internet_service='DSL',
            online_security='Yes',
            online_backup='Yes',
            device_protection='Yes',
            tech_support='Yes',
            streaming_tv='Yes',
            streaming_movies='Yes',
            contract_type='Two year',
            paperless_billing=False,
            payment_method='Bank transfer (automatic)',
            monthly_charges=45.99,
            total_charges=1200.50,
            tenure_months=48
        )
        
        risk_score = self.service.calculate_risk_score(low_risk_customer)
        
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0
        # Low-risk indicators: long-term contract, lower monthly charges, more services
        assert risk_score < 0.5
    
    def test_get_customer_segment_premium(self):
        """Test customer segment classification for premium customers."""
        premium_customer = Customer(
            customer_id='CUST-003',
            gender='Female',
            senior_citizen=False,
            partner=True,
            dependents=True,
            phone_service=True,
            multiple_lines='Yes',
            internet_service='Fiber optic',
            online_security='Yes',
            online_backup='Yes',
            device_protection='Yes',
            tech_support='Yes',
            streaming_tv='Yes',
            streaming_movies='Yes',
            contract_type='Two year',
            paperless_billing=True,
            payment_method='Credit card (automatic)',
            monthly_charges=89.99,
            total_charges=2000.75,
            tenure_months=36
        )
        
        segment = self.service.get_customer_segment(premium_customer)
        
        assert segment == 'Premium'
    
    def test_get_customer_segment_basic(self):
        """Test customer segment classification for basic customers."""
        basic_customer = Customer(
            customer_id='CUST-004',
            gender='Male',
            senior_citizen=False,
            partner=False,
            dependents=False,
            phone_service=True,
            multiple_lines='No',
            internet_service='DSL',
            online_security='No',
            online_backup='No',
            device_protection='No',
            tech_support='No',
            streaming_tv='No',
            streaming_movies='No',
            contract_type='Month-to-month',
            paperless_billing=False,
            payment_method='Mailed check',
            monthly_charges=29.99,
            total_charges=400.25,
            tenure_months=12
        )
        
        segment = self.service.get_customer_segment(basic_customer)
        
        assert segment == 'Basic'
    
    def test_generate_recommendations_high_risk(self):
        """Test recommendation generation for high-risk customers."""
        recommendations = self.service.generate_recommendations(
            self.sample_customer, 
            churn_probability=0.85,
            risk_level='High'
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should include retention strategies for high-risk customers
        assert any('retention' in rec.lower() or 'discount' in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_low_risk(self):
        """Test recommendation generation for low-risk customers."""
        recommendations = self.service.generate_recommendations(
            self.sample_customer,
            churn_probability=0.15,
            risk_level='Low'
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should include upselling/cross-selling strategies
        assert any('upgrade' in rec.lower() or 'additional' in rec.lower() for rec in recommendations)


class TestCsvCustomerRepository:
    """Test cases for CsvCustomerRepository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_csv_path = 'test_customers.csv'
        self.repository = CsvCustomerRepository(self.test_csv_path)
        self.sample_customer = Customer(
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
    
    def teardown_method(self):
        """Clean up test files."""
        import os
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    @patch('pandas.read_csv')
    def test_load_customers(self, mock_read_csv):
        """Test loading customers from CSV."""
        # Mock CSV data
        mock_data = pd.DataFrame({
            'customer_id': ['CUST-001', 'CUST-002'],
            'gender': ['Female', 'Male'],
            'senior_citizen': [False, True],
            'partner': [True, False],
            'dependents': [False, True],
            'phone_service': [True, False],
            'multiple_lines': ['No', 'No'],
            'internet_service': ['Fiber optic', 'DSL'],
            'online_security': ['No', 'Yes'],
            'online_backup': ['Yes', 'No'],
            'device_protection': ['No', 'Yes'],
            'tech_support': ['No', 'Yes'],
            'streaming_tv': ['Yes', 'No'],
            'streaming_movies': ['No', 'No'],
            'contract_type': ['Month-to-month', 'Two year'],
            'paperless_billing': [True, False],
            'payment_method': ['Electronic check', 'Bank transfer (automatic)'],
            'monthly_charges': [70.35, 45.99],
            'total_charges': [1510.45, 1200.50],
            'tenure_months': [24, 48]
        })
        mock_read_csv.return_value = mock_data
        
        customers = self.repository.load_customers()
        
        assert len(customers) == 2
        assert customers[0].customer_id == 'CUST-001'
        assert customers[1].customer_id == 'CUST-002'
        mock_read_csv.assert_called_once()
    
    def test_save_customer(self):
        """Test saving a single customer."""
        self.repository.save_customer(self.sample_customer)
        
        # Verify file was created
        import os
        assert os.path.exists(self.test_csv_path)
        
        # Verify data was saved correctly
        df = pd.read_csv(self.test_csv_path)
        assert len(df) == 1
        assert df.iloc[0]['customer_id'] == 'CUST-001'
        assert df.iloc[0]['gender'] == 'Female'
        assert df.iloc[0]['monthly_charges'] == 70.35
    
    def test_save_customers(self):
        """Test saving multiple customers."""
        customers = [
            self.sample_customer,
            Customer(
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
        
        self.repository.save_customers(customers)
        
        # Verify data was saved correctly
        df = pd.read_csv(self.test_csv_path)
        assert len(df) == 2
        assert df.iloc[0]['customer_id'] == 'CUST-001'
        assert df.iloc[1]['customer_id'] == 'CUST-002'
    
    def test_find_by_id(self):
        """Test finding customer by ID."""
        # Save a customer first
        self.repository.save_customer(self.sample_customer)
        
        # Find the customer
        found_customer = self.repository.find_by_id('CUST-001')
        
        assert found_customer is not None
        assert found_customer.customer_id == 'CUST-001'
        assert found_customer.gender == 'Female'
        assert found_customer.monthly_charges == 70.35
    
    def test_find_by_id_not_found(self):
        """Test finding non-existent customer."""
        customer = self.repository.find_by_id('NON-EXISTENT')
        assert customer is None
    
    def test_update_customer(self):
        """Test updating customer information."""
        # Save initial customer
        self.repository.save_customer(self.sample_customer)
        
        # Update customer data
        updated_customer = Customer(
            customer_id='CUST-001',
            gender='Female',
            senior_citizen=False,
            partner=True,
            dependents=False,
            phone_service=True,
            multiple_lines='Yes',  # Changed from 'No'
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
            monthly_charges=75.99,  # Changed from 70.35
            total_charges=1600.45,  # Changed from 1510.45
            tenure_months=30  # Changed from 24
        )
        
        result = self.repository.update_customer('CUST-001', updated_customer)
        
        assert result is True
        
        # Verify update
        df = pd.read_csv(self.test_csv_path)
        assert df.iloc[0]['multiple_lines'] == 'Yes'
        assert df.iloc[0]['monthly_charges'] == 75.99
        assert df.iloc[0]['tenure_months'] == 30
    
    def test_delete_customer(self):
        """Test deleting customer."""
        # Save customer first
        self.repository.save_customer(self.sample_customer)
        
        # Delete customer
        result = self.repository.delete_customer('CUST-001')
        
        assert result is True
        
        # Verify deletion
        df = pd.read_csv(self.test_csv_path)
        assert len(df) == 0
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        # Save multiple customers
        customers = [
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
            ) for i in range(5)
        ]
        
        self.repository.save_customers(customers)
        
        stats = self.repository.get_summary_statistics()
        
        assert 'total_customers' in stats
        assert 'avg_monthly_charges' in stats
        assert 'avg_tenure_months' in stats
        assert stats['total_customers'] == 5
        assert stats['avg_monthly_charges'] > 0
        assert stats['avg_tenure_months'] > 0


class TestSklearnChurnPredictionModel:
    """Test cases for SklearnChurnPredictionModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_path = 'test_model.pkl'
        self.preprocessor_path = 'test_preprocessor.pkl'
        self.model = SklearnChurnPredictionModel(self.model_path, self.preprocessor_path)
        
        self.sample_customer = Customer(
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
    
    def teardown_method(self):
        """Clean up test files."""
        import os
        for path in [self.model_path, self.preprocessor_path]:
            if os.path.exists(path):
                os.remove(path)
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_joblib_load):
        """Test successful model loading."""
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_joblib_load.side_effect = [mock_model, mock_preprocessor]
        
        result = self.model.load_model()
        
        assert result is True
        assert self.model.model == mock_model
        assert self.model.preprocessor == mock_preprocessor
    
    @patch('joblib.load')
    def test_load_model_failure(self, mock_joblib_load):
        """Test model loading failure."""
        mock_joblib_load.side_effect = FileNotFoundError("Model file not found")
        
        result = self.model.load_model()
        
        assert result is False
    
    @patch('joblib.load')
    def test_predict_single(self, mock_joblib_load):
        """Test single customer prediction."""
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 70.35, 1510.45, 24]])
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        mock_joblib_load.side_effect = [mock_model, mock_preprocessor]
        
        self.model.load_model()
        result = self.model.predict_single(self.sample_customer)
        
        assert 'churn_probability' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        assert result['churn_probability'] == 0.85
        assert result['confidence'] > 0
        assert result['risk_level'] in ['Low', 'Medium', 'High']
    
    @patch('joblib.load')
    def test_predict_batch(self, mock_joblib_load):
        """Test batch prediction."""
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 70.35, 1510.45, 24],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 45.99, 1200.50, 48]
        ])
        mock_model.predict_proba.return_value = np.array([
            [0.15, 0.85],
            [0.80, 0.20]
        ])
        mock_joblib_load.side_effect = [mock_model, mock_preprocessor]
        
        customers = [self.sample_customer, Customer(
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
        )]
        
        self.model.load_model()
        results = self.model.predict_batch(customers)
        
        assert len(results) == 2
        assert results[0]['churn_probability'] == 0.85
        assert results[1]['churn_probability'] == 0.20
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        confidence = self.model._calculate_confidence(0.85)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Higher probability should result in higher confidence
        assert confidence > 0.5
    
    def test_determine_risk_level(self):
        """Test risk level determination."""
        # Test high risk
        risk_level = self.model._determine_risk_level(0.85)
        assert risk_level == 'High'
        
        # Test medium risk
        risk_level = self.model._determine_risk_level(0.55)
        assert risk_level == 'Medium'
        
        # Test low risk
        risk_level = self.model._determine_risk_level(0.25)
        assert risk_level == 'Low'
    
    def test_customer_to_model_input(self):
        """Test customer data conversion to model input format."""
        input_data = self.model._customer_to_model_input(self.sample_customer)
        
        assert isinstance(input_data, pd.DataFrame)
        assert len(input_data.columns) > 0
        assert input_data.iloc[0]['tenure_months'] == 24
        assert input_data.iloc[0]['monthly_charges'] == 70.35
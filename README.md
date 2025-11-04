# Customer Churn Prediction System

A comprehensive machine learning system for predicting customer churn using hexagonal architecture (clean architecture) principles. This project implements a production-ready solution with FastAPI, Streamlit, Airflow, and Docker.

## 🎯 Project Overview

This system predicts customer churn using machine learning models and provides both programmatic and user-friendly interfaces for making predictions. The architecture follows hexagonal (ports and adapters) design principles to ensure maintainability, testability, and scalability.

### Key Features

- **Machine Learning Pipeline**: Automated data preprocessing, model training, and evaluation
- **RESTful API**: FastAPI-based service for programmatic access
- **Web Interface**: Streamlit-based dashboard for interactive predictions
- **Automated Training**: Airflow DAGs for scheduled model retraining
- **Containerized Deployment**: Docker support for easy deployment
- **Comprehensive Testing**: Unit and integration tests
- **Monitoring**: Health checks and logging

## 🏗️ Architecture

The project follows **Hexagonal Architecture** (Clean Architecture) principles:

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │   FastAPI   │  │  Streamlit  │  │     Airflow        │  │
│  │   (REST)    │  │   (Web UI)  │  │    (Scheduler)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬─────────┘  │
│         │                │                      │            │
├─────────┼────────────────┼──────────────────────┼────────────┤
│         │                │                      │            │
│         ▼                ▼                      ▼            │
│                    Application Layer                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Use Cases (Orchestration)                │  │
│  │  • PredictCustomerChurnUseCase                      │  │
│  │  • TrainChurnModelUseCase                            │  │
│  │  • ProcessCustomerBatchUseCase                       │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                     │
├───────────────────────┼─────────────────────────────────────┤
│                       │                                     │
│                       ▼                                     │
│                    Domain Layer                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Business Logic & Entities                │  │
│  │  • Customer (Entity)                                  │  │
│  │  • ChurnAnalysisService (Business Rules)              │  │
│  │  • ChurnPredictionModel (Interface)                   │  │
│  │  • CustomerRepository (Interface)                     │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                     │
├───────────────────────┼─────────────────────────────────────┤
│                       │                                     │
│                       ▼                                     │
│                    Adapter Layer                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              External Integrations                     │  │
│  │  • CsvCustomerRepository (Data Access)                │  │
│  │  • SklearnChurnPredictionModel (ML Framework)       │  │
│  │  • External APIs, Databases, etc.                   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

1. **Domain Layer**: Core business logic, entities, and domain services
2. **Application Layer**: Use cases that orchestrate domain objects
3. **Adapter Layer**: External integrations (databases, ML frameworks, APIs)
4. **Interface Layer**: User interfaces (REST API, Web UI, CLI)

## 📁 Project Structure

```
proyecto_clasificacion_churn_hexagonal/
├── src/
│   ├── domain/                    # Domain layer (core business logic)
│   │   ├── entities/              # Business entities (Customer)
│   │   ├── repositories/          # Repository interfaces
│   │   └── services/              # Domain services
│   ├── application/               # Application layer (use cases)
│   │   ├── dto/                   # Data Transfer Objects
│   │   └── use_cases/             # Application use cases
│   ├── adapters/                  # Adapter layer (external integrations)
│   │   ├── repositories/          # Repository implementations
│   │   └── machine_learning/       # ML model implementations
│   ├── interface/                 # Interface layer (user interfaces)
│   │   ├── api/                   # FastAPI REST endpoints
│   │   └── web/                   # Streamlit web interface
│   ├── dags/                      # Airflow DAGs
│   └── configs/                   # Configuration files
├── docker/                        # Docker configuration
├── data/                          # Data directory
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data
├── models/                        # Trained models
├── tests/                         # Test files
└── logs/                          # Log files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/FrankYesid/proyecto_clasificacion_churn_hexagonal.git
cd proyecto_clasificacion_churn_hexagonal
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env file with your configuration
```

Key environment variables:
- `API_BASE_URL`: Base URL for the API service
- `MODEL_PATH`: Path to the trained model file
- `DB_*`: Database configuration (if using)
- `SMTP_*`: Email configuration for notifications

### 4. Run with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

This will start:
- **API Service**: http://localhost:8000
- **Web Interface**: http://localhost:8501
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 5. Manual Setup (Alternative)

If you prefer to run components individually:

#### Start API Service
```bash
uvicorn src.interface.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Start Web Interface
```bash
streamlit run src/interface/web/streamlit_app.py --server.port 8501
```

#### Start Airflow
```bash
# Initialize database
airflow db init

# Create admin user
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# Start webserver
airflow webserver --port 8080

# Start scheduler (in another terminal)
airflow scheduler
```

## 📊 Usage

### API Endpoints

The API provides the following endpoints:

#### Health Check
```bash
GET /health
```

#### Model Information
```bash
GET /model/info
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

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

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "batch_id": "BATCH-001",
  "customers": [
    { /* customer data */ },
    { /* customer data */ }
  ]
}
```

#### CSV File Prediction
```bash
POST /predict/csv
Content-Type: multipart/form-data

file: <upload CSV file>
```

### Web Interface

Access the Streamlit web interface at http://localhost:8501

Features:
- **Single Prediction**: Enter customer details manually
- **Batch Prediction**: Upload CSV files or use sample data
- **Analytics Dashboard**: Visualize prediction results
- **Model Information**: View model details and performance

### Airflow DAGs

Access the Airflow UI at http://localhost:8080

Available DAGs:
- **churn_model_training_pipeline**: Automated weekly model retraining

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_customer_service.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── domain/             # Domain layer tests
│   ├── application/        # Application layer tests
│   └── adapters/           # Adapter layer tests
├── integration/             # Integration tests
│   ├── test_api.py         # API integration tests
│   └── test_repositories.py # Repository integration tests
└── fixtures/               # Test data and fixtures
```

## 📈 Model Performance

The current model achieves the following performance metrics:

- **Accuracy**: 85.3%
- **Precision**: 80.1%
- **Recall**: 75.8%
- **F1-Score**: 77.9%
- **ROC-AUC**: 0.87

### Feature Importance

Top features contributing to churn prediction:
1. Contract type (Month-to-month vs Long-term)
2. Tenure months
3. Payment method
4. Monthly charges
5. Internet service type

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_ENV` | Environment (development/production) | `development` |
| `API_PORT` | API server port | `8000` |
| `MODEL_PATH` | Path to trained model | `models/latest_model.pkl` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DB_HOST` | Database host | `localhost` |
| `SMTP_HOST` | Email SMTP host | `smtp.gmail.com` |

### Model Configuration

The model can be configured through environment variables:

- `TRAINING_TEST_SIZE`: Train/test split ratio (default: 0.2)
- `TRAINING_RANDOM_STATE`: Random seed (default: 42)
- `MIN_MODEL_ACCURACY`: Minimum accuracy threshold (default: 0.75)
- `MIN_MODEL_PRECISION`: Minimum precision threshold (default: 0.70)
- `MIN_MODEL_RECALL`: Minimum recall threshold (default: 0.65)

## 🐳 Docker Configuration

### Services

The `docker-compose.yml` defines the following services:

1. **api**: FastAPI service
2. **web**: Streamlit web interface
3. **postgres**: PostgreSQL database
4. **airflow-webserver**: Airflow web interface
5. **airflow-scheduler**: Airflow scheduler
6. **redis**: Redis cache (optional)
7. **log-monitor**: Log monitoring (optional)

### Building Custom Images

```bash
# Build API image
docker build -f docker/Dockerfile.api -t churn-api:latest .

# Build web image
docker build -f docker/Dockerfile.web -t churn-web:latest .
```

## 📊 Data Format

### Customer Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | string | Unique customer identifier |
| `gender` | string | Customer gender (Male/Female) |
| `senior_citizen` | boolean | Senior citizen status |
| `partner` | boolean | Has partner |
| `dependents` | boolean | Has dependents |
| `phone_service` | boolean | Phone service subscription |
| `multiple_lines` | string | Multiple lines status |
| `internet_service` | string | Internet service type |
| `online_security` | string | Online security service |
| `online_backup` | string | Online backup service |
| `device_protection` | string | Device protection service |
| `tech_support` | string | Tech support service |
| `streaming_tv` | string | Streaming TV service |
| `streaming_movies` | string | Streaming movies service |
| `contract_type` | string | Contract type |
| `paperless_billing` | boolean | Paperless billing |
| `payment_method` | string | Payment method |
| `monthly_charges` | float | Monthly charges amount |
| `total_charges` | float | Total charges amount |
| `tenure_months` | integer | Tenure in months |

### Sample Data

Example customer data:

```json
{
  "customer_id": "CUST-12345",
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

## 🔒 Security

### API Security

- JWT-based authentication (configurable)
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection

### Data Security

- Data encryption at rest
- Secure data transmission (HTTPS)
- PII data masking
- Access control

### Best Practices

- Regular security updates
- Dependency scanning
- Security headers
- CORS configuration
- Environment variable protection

## 📈 Monitoring and Logging

### Health Checks

The API provides health check endpoints:

```bash
GET /health
GET /health/detailed
```

### Logging

Structured logging with different levels:

- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Metrics

Available metrics:

- Request count and latency
- Model prediction accuracy
- Error rates
- Resource utilization

### Monitoring Tools

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation
- **Alertmanager**: Alert notifications

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   export API_ENV=production
   export MODEL_PATH=/app/models/production_model.pkl
   export LOG_LEVEL=WARNING
   ```

2. **Database Migration**
   ```bash
   # Run database migrations if using database
   python scripts/migrate_db.py
   ```

3. **Model Deployment**
   ```bash
   # Deploy trained model to production
   python scripts/deploy_model.py --model-path models/latest_model.pkl
   ```

4. **Service Start**
   ```bash
   # Start services with production configuration
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Scaling

For high-traffic scenarios:

1. **Horizontal Scaling**
   - Use load balancers
   - Deploy multiple API instances
   - Implement service discovery

2. **Vertical Scaling**
   - Increase CPU/memory resources
   - Optimize model inference
   - Use GPU acceleration

3. **Caching**
   - Implement Redis caching
   - Use CDN for static assets
   - Cache prediction results

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints
- Write docstrings

### Testing Guidelines

- Write unit tests for new features
- Maintain test coverage above 80%
- Include integration tests
- Test edge cases

## 📚 Documentation

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Code Documentation

Generate code documentation:

```bash
# Install documentation dependencies
pip install sphinx

# Generate documentation
cd docs
make html
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   netstat -ano | findstr :8000
   # Kill process
   taskkill /PID <PID> /F
   ```

2. **Docker Build Fails**
   ```bash
   # Clear Docker cache
   docker system prune -a
   # Rebuild images
   docker-compose build --no-cache
   ```

3. **Model Loading Error**
   ```bash
   # Check model file exists
   ls -la models/
   # Verify model format
   python -c "import joblib; joblib.load('models/latest_model.pkl')"
   ```

4. **Database Connection Error**
   ```bash
   # Check PostgreSQL status
   docker-compose ps postgres
   # Check connection
   docker-compose exec postgres psql -U airflow -d airflow
   ```

### Performance Issues

1. **Slow Predictions**
   - Check model size and complexity
   - Optimize feature engineering
   - Consider model quantization

2. **High Memory Usage**
   - Monitor memory consumption
   - Implement batch processing
   - Use memory-efficient data structures

3. **API Latency**
   - Enable response caching
   - Optimize database queries
   - Use connection pooling

## 📞 Support

For support and questions:

- Create an issue in the GitHub repository
- Check the documentation
- Review the troubleshooting section

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team for the excellent ML library
- FastAPI team for the modern web framework
- Streamlit team for the intuitive web app framework
- Apache Airflow team for the workflow orchestration platform

## 📈 Roadmap

### Short-term (Next 3 months)

- [ ] Add support for additional ML models
- [ ] Implement real-time model monitoring
- [ ] Add A/B testing capabilities
- [ ] Enhance data visualization

### Medium-term (Next 6 months)

- [ ] Add support for deep learning models
- [ ] Implement federated learning
- [ ] Add multi-language support
- [ ] Integrate with cloud ML platforms

### Long-term (Next 12 months)

- [ ] Implement autoML capabilities
- [ ] Add explainable AI features
- [ ] Support for edge deployment
- [ ] Integration with IoT devices

---

**Made with ❤️ by the ML Team**
"""
Aplicación Streamlit para la interfaz web de predicción de churn.
Interfaz interactiva para cargar datos y visualizar predicciones.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import io

# Configuración de la página
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class ChurnPredictionApp:
    """
    Clase principal de la aplicación Streamlit.
    """
    
    def __init__(self):
        """
        Inicializa la aplicación.
        """
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()
        
        # Inicializar estado de la sesión
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'api_status' not in st.session_state:
            st.session_state.api_status = None
    
    def run(self):
        """
        Ejecuta la aplicación.
        """
        # Header principal
        st.markdown('<div class="main-header">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Verificar estado de la API
        self.check_api_status()
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Single Prediction", 
            "📁 Batch Prediction", 
            "📊 Analytics", 
            "ℹ️ Info"
        ])
        
        with tab1:
            self.render_single_prediction()
        
        with tab2:
            self.render_batch_prediction()
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_info()
    
    def render_sidebar(self):
        """
        Renderiza el sidebar con configuraciones.
        """
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Configuración de API
            st.subheader("API Settings")
            api_url = st.text_input(
                "API Base URL",
                value=self.api_base_url,
                help="URL base del servicio API"
            )
            self.api_base_url = api_url
            
            # Configuración de visualización
            st.subheader("Display Settings")
            self.show_confidence = st.checkbox("Show Confidence Scores", value=True)
            self.show_risk_scores = st.checkbox("Show Risk Scores", value=True)
            self.show_segments = st.checkbox("Show Customer Segments", value=True)
            
            # Información del modelo
            if st.session_state.api_status:
                st.subheader("Model Info")
                model_info = st.session_state.api_status.get('model_info', {})
                if model_info:
                    st.write(f"**Version:** {model_info.get('model_version', 'N/A')}")
                    st.write(f"**Type:** {model_info.get('model_type', 'N/A')}")
                    st.write(f"**Status:** {model_info.get('status', 'N/A')}")
            
            # Acerca de
            st.divider()
            st.markdown("""
            **Customer Churn Prediction App**
            
            *Powered by Machine Learning*
            
            This application predicts customer churn using a trained ML model.
            Upload customer data or enter individual customer information
to get churn predictions with detailed analysis.
            """)
    
    def check_api_status(self):
        """
        Verifica el estado de la API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.api_status = response.json()
                
                # Obtener información del modelo
                try:
                    model_response = self.session.get(f"{self.api_base_url}/model/info", timeout=5)
                    if model_response.status_code == 200:
                        st.session_state.api_status['model_info'] = model_response.json()
                except:
                    pass
            else:
                st.session_state.api_status = None
        except requests.exceptions.RequestException:
            st.session_state.api_status = None
    
    def render_single_prediction(self):
        """
        Renderiza la interfaz de predicción individual.
        """
        st.header("🎯 Single Customer Prediction")
        
        if st.session_state.api_status is None:
            st.error("❌ API service is not available. Please check the API URL in the sidebar.")
            return
        
        # Formulario de entrada
        with st.form("single_prediction_form"):
            st.subheader("Customer Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id = st.text_input("Customer ID", value="CUST-001")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.checkbox("Senior Citizen")
                partner = st.checkbox("Has Partner")
                dependents = st.checkbox("Has Dependents")
                phone_service = st.checkbox("Phone Service")
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            
            with col2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.checkbox("Paperless Billing")
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            
            # Campos numéricos
            st.subheader("Service Details")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=240, value=24)
            
            with col4:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0, step=0.01)
            
            with col5:
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0, step=0.01)
            
            # Botón de predicción
            submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)
        
        # Realizar predicción
        if submitted:
            with st.spinner("Analyzing customer..."):
                customer_data = {
                    "customer_id": customer_id,
                    "gender": gender,
                    "senior_citizen": senior_citizen,
                    "partner": partner,
                    "dependents": dependents,
                    "phone_service": phone_service,
                    "multiple_lines": multiple_lines,
                    "internet_service": internet_service,
                    "online_security": online_security,
                    "online_backup": online_backup,
                    "device_protection": device_protection,
                    "tech_support": tech_support,
                    "streaming_tv": streaming_tv,
                    "streaming_movies": streaming_movies,
                    "contract_type": contract_type,
                    "paperless_billing": paperless_billing,
                    "payment_method": payment_method,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "tenure_months": tenure_months
                }
                
                try:
                    response = self.session.post(
                        f"{self.api_base_url}/predict",
                        json=customer_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        self.display_prediction_result(prediction)
                    else:
                        st.error(f"Prediction failed: {response.status_code} - {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
    
    def render_batch_prediction(self):
        """
        Renderiza la interfaz de predicción en lote.
        """
        st.header("📁 Batch Prediction")
        
        if st.session_state.api_status is None:
            st.error("❌ API service is not available.")
            return
        
        # Opciones de entrada
        input_method = st.radio("Choose input method:", ["Upload CSV File", "Use Sample Data"])
        
        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"✅ File uploaded successfully! {len(df)} customers loaded.")
                    
                    # Vista previa de datos
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10))
                        st.write(f"**Total customers:** {len(df)}")
                        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                    
                    # Botón de predicción
                    if st.button("🔮 Predict All Customers", type="primary"):
                        self.predict_batch_from_dataframe(df)
                        
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        else:  # Sample Data
            st.info("Using sample customer data for demonstration.")
            
            # Crear datos de muestra
            sample_data = self.create_sample_data()
            st.session_state.uploaded_data = sample_data
            
            with st.expander("📋 Sample Data Preview"):
                st.dataframe(sample_data.head(10))
                st.write(f"**Total sample customers:** {len(sample_data)}")
            
            if st.button("🔮 Predict Sample Customers", type="primary"):
                self.predict_batch_from_dataframe(sample_data)
        
        # Mostrar resultados si existen
        if st.session_state.predictions is not None:
            self.display_batch_results(st.session_state.predictions)
    
    def render_analytics(self):
        """
        Renderiza el panel de análisis y visualizaciones.
        """
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.predictions is None:
            st.info("ℹ️ No prediction data available. Please run a batch prediction first.")
            return
        
        predictions = st.session_state.predictions
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(predictions['results'])
            st.metric("Total Customers", total_customers)
        
        with col2:
            churn_predictions = sum(1 for r in predictions['results'] if r['prediction'])
            churn_rate = (churn_predictions / total_customers) * 100 if total_customers > 0 else 0
            st.metric("Predicted Churn", f"{churn_rate:.1f}%")
        
        with col3:
            avg_confidence = np.mean([r['confidence'] for r in predictions['results']])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            high_risk = sum(1 for r in predictions['results'] if r['risk_level'] in ['High', 'Critical'])
            st.metric("High Risk", high_risk)
        
        # Distribución de predicciones
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de pastel - Predicciones
            churn_counts = [churn_predictions, total_customers - churn_predictions]
            labels = ['Churn', 'No Churn']
            colors = ['#d62728', '#2ca02c']
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=churn_counts, marker_colors=colors)])
            fig_pie.update_layout(title="Churn Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gráfico de barras - Niveles de riesgo
            risk_levels = [r['risk_level'] for r in predictions['results']]
            risk_counts = pd.Series(risk_levels).value_counts()
            
            fig_bar = px.bar(
                x=risk_counts.index, 
                y=risk_counts.values,
                title="Risk Level Distribution",
                labels={'x': 'Risk Level', 'y': 'Count'},
                color=risk_counts.index,
                color_discrete_map={'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Análisis por segmentos
        st.subheader("Customer Segment Analysis")
        
        segments = [r['customer_segment'] for r in predictions['results']]
        segment_counts = pd.Series(segments).value_counts()
        
        fig_segments = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title="Customer Segments",
            labels={'x': 'Segment', 'y': 'Count'},
            color=segment_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Distribución de probabilidades
        st.subheader("Probability Distribution")
        
        probabilities = [r['probability_churn'] for r in predictions['results']]
        
        fig_hist = px.histogram(
            x=probabilities,
            nbins=20,
            title="Churn Probability Distribution",
            labels={'x': 'Churn Probability', 'y': 'Count'}
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tabla de resultados detallados
        st.subheader("Detailed Results")
        
        results_df = pd.DataFrame(predictions['results'])
        
        # Columnas a mostrar
        display_columns = ['customer_id', 'prediction', 'probability_churn', 'confidence', 'risk_level', 'customer_segment']
        if self.show_risk_scores:
            display_columns.append('risk_score')
        if self.show_confidence:
            display_columns.append('confidence')
        if self.show_segments:
            display_columns.append('customer_segment')
        
        # Filtrar por riesgo
        risk_filter = st.multiselect(
            "Filter by Risk Level:",
            ['Low', 'Medium', 'High', 'Critical'],
            default=['Low', 'Medium', 'High', 'Critical']
        )
        
        filtered_df = results_df[results_df['risk_level'].isin(risk_filter)]
        
        # Mostrar tabla con formato condicional
        st.dataframe(
            filtered_df[display_columns].style.apply(self.color_risk_level, axis=1),
            use_container_width=True
        )
        
        # Descargar resultados
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_info(self):
        """
        Renderiza la pestaña de información.
        """
        st.header("ℹ️ Information")
        
        # Estado de la API
        st.subheader("API Status")
        if st.session_state.api_status:
            status = st.session_state.api_status
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {status.get('status', 'Unknown')}")
                st.write(f"**Model Status:** {status.get('model_status', 'Unknown')}")
                st.write(f"**Uptime:** {status.get('uptime_seconds', 0):.0f} seconds")
            
            with col2:
                if 'model_info' in status:
                    model_info = status['model_info']
                    st.write(f"**Model Version:** {model_info.get('model_version', 'N/A')}")
                    st.write(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
                    st.write(f"**Features:** {len(model_info.get('features', []))}")
        else:
            st.error("API service is not available.")
        
        # Guía de uso
        st.subheader("📖 How to Use")
        
        st.markdown("""
        ### Single Prediction
        1. Go to the **Single Prediction** tab
        2. Fill in the customer information form
        3. Click **Predict Churn** to get the prediction
        4. View the detailed analysis and recommendations
        
        ### Batch Prediction
        1. Go to the **Batch Prediction** tab
        2. Choose to upload a CSV file or use sample data
        3. Click **Predict All Customers** to process the batch
        4. View results in the **Analytics** tab
        
        ### Analytics Dashboard
        The dashboard provides:
        - Churn prediction distribution
        - Risk level analysis
        - Customer segmentation
        - Probability distributions
        - Detailed results table
        
        ### CSV File Format
        Your CSV file should contain the following columns:
        - customer_id, gender, senior_citizen, partner, dependents
        - phone_service, multiple_lines, internet_service
        - online_security, online_backup, device_protection, tech_support
        - streaming_tv, streaming_movies, contract_type, paperless_billing
        - payment_method, monthly_charges, total_charges, tenure_months
        """)
        
        # Información del modelo
        st.subheader("🤖 Model Information")
        st.markdown("""
        This application uses a **Random Forest Classifier** model trained on historical customer data.
        
        **Features:**
        - Customer demographics (age, gender, location)
        - Service usage patterns
        - Billing information
        - Contract details
        - Support interactions
        
        **Model Performance:**
        - Accuracy: ~85%
        - Precision: ~80%
        - Recall: ~75%
        - F1-Score: ~77%
        
        **Risk Levels:**
        - **Low:** Probability < 0.4
        - **Medium:** Probability 0.4-0.6
        - **High:** Probability 0.6-0.8
        - **Critical:** Probability > 0.8
        """)
    
    def display_prediction_result(self, prediction: Dict[str, Any]):
        """
        Muestra el resultado de una predicción individual.
        """
        # Tarjeta de resultado principal
        st.success("✅ Prediction completed successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Predicción principal
            if prediction['prediction']:
                st.error(f"🚨 **CHURN PREDICTED**")
                st.write(f"**Risk Level:** <span class='risk-high'>{prediction['risk_level']}</span>", unsafe_allow_html=True)
            else:
                st.success(f"✅ **NO CHURN PREDICTED**")
                st.write(f"**Risk Level:** <span class='risk-low'>{prediction['risk_level']}</span>", unsafe_allow_html=True)
            
            st.write(f"**Confidence:** {prediction['confidence']:.2%}")
            st.write(f"**Churn Probability:** {prediction['probability_churn']:.2%}")
        
        with col2:
            # Métricas adicionales
            st.write(f"**Customer Segment:** {prediction['customer_segment']}")
            st.write(f"**Risk Score:** {prediction['risk_score']:.2f}")
            st.write(f"**Lifetime Value:** ${prediction['lifetime_value']:,.2f}")
            st.write(f"**Model Version:** {prediction['model_version']}")
        
        # Visualización de probabilidades
        st.subheader("Probability Breakdown")
        
        prob_data = pd.DataFrame({
            'Outcome': ['Churn', 'Stay'],
            'Probability': [prediction['probability_churn'], prediction['probability_stay']]
        })
        
        fig = px.bar(
            prob_data, 
            x='Outcome', 
            y='Probability',
            title="Churn vs Stay Probability",
            color='Outcome',
            color_discrete_map={'Churn': '#d62728', 'Stay': '#2ca02c'}
        )
        fig.update_yaxis(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendaciones
        self.display_recommendations(prediction)
    
    def predict_batch_from_dataframe(self, df: pd.DataFrame):
        """
        Realiza predicciones en lote desde un DataFrame.
        """
        try:
            # Preparar datos para la API
            customers = []
            for _, row in df.iterrows():
                customer_data = self.row_to_customer_dict(row)
                customers.append(customer_data)
            
            batch_data = {
                "batch_id": f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "customers": customers
            }
            
            # Realizar predicción
            with st.spinner(f"Processing {len(customers)} customers..."):
                response = self.session.post(
                    f"{self.api_base_url}/predict/batch",
                    json=batch_data,
                    timeout=300  # 5 minutos timeout para lotes grandes
                )
                
                if response.status_code == 200:
                    predictions = response.json()
                    st.session_state.predictions = predictions
                    st.success(f"✅ Batch prediction completed! Processed {predictions['processed_count']} customers.")
                    
                    if predictions['error_count'] > 0:
                        st.warning(f"⚠️ {predictions['error_count']} customers had errors during processing.")
                else:
                    st.error(f"Batch prediction failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"Error during batch prediction: {str(e)}")
    
    def display_batch_results(self, predictions: Dict[str, Any]):
        """
        Muestra los resultados de predicción en lote.
        """
        st.success("✅ Batch prediction results:")
        
        # Resumen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processed", predictions['processed_count'])
        
        with col2:
            churn_count = sum(1 for r in predictions['results'] if r['prediction'])
            churn_rate = (churn_count / predictions['processed_count']) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        with col3:
            st.metric("Processing Time", f"{predictions['processing_time_seconds']:.1f}s")
        
        # Tabla de resultados
        results_df = pd.DataFrame(predictions['results'])
        
        st.subheader("Prediction Results")
        
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_filter = st.multiselect(
                "Filter by Prediction:",
                ['Churn', 'No Churn'],
                default=['Churn', 'No Churn']
            )
        
        with col2:
            risk_filter = st.multiselect(
                "Filter by Risk Level:",
                ['Low', 'Medium', 'High', 'Critical'],
                default=['Low', 'Medium', 'High', 'Critical']
            )
        
        # Aplicar filtros
        filtered_df = results_df[
            (results_df['prediction'].isin([p == 'Churn' for p in prediction_filter])) &
            (results_df['risk_level'].isin(risk_filter))
        ]
        
        # Mostrar tabla con formato
        st.dataframe(
            filtered_df.style.apply(self.color_risk_level, axis=1),
            use_container_width=True
        )
        
        # Estadísticas adicionales
        if st.checkbox("Show detailed statistics"):
            self.show_detailed_statistics(results_df)
    
    def display_recommendations(self, prediction: Dict[str, Any]):
        """
        Muestra recomendaciones basadas en la predicción.
        """
        st.subheader("📋 Recommendations")
        
        if prediction['prediction']:  # Churn predicho
            st.warning("**High Risk Customer - Immediate Action Required:**")
            
            recommendations = []
            
            if prediction['risk_level'] in ['High', 'Critical']:
                recommendations.extend([
                    "🎯 **Immediate outreach** - Contact customer within 24 hours",
                    "💰 **Retention offer** - Prepare personalized discount or upgrade",
                    "📞 **Personal call** - Schedule call with customer success team"
                ])
            
            if prediction['probability_churn'] > 0.7:
                recommendations.append("🔥 **Priority intervention** - This customer has very high churn probability")
            
            if prediction['customer_segment'] == 'High Value High Risk':
                recommendations.append("💎 **VIP treatment** - High-value customer requiring special attention")
            
            recommendations.extend([
                "📊 **Analyze usage patterns** - Review customer's service usage",
                "📝 **Survey feedback** - Collect feedback on service satisfaction",
                "🛠️ **Technical support** - Offer proactive technical assistance"
            ])
            
            for rec in recommendations:
                st.write(rec)
        
        else:  # No churn predicho
            st.success("**Low Risk Customer - Maintenance Strategy:**")
            
            recommendations = [
                "✅ **Continue excellent service** - Maintain current service quality",
                "🎁 **Loyalty rewards** - Consider loyalty program or small perks",
                "📈 **Upselling opportunity** - Customer is stable, consider service upgrades",
                "🔄 **Regular check-ins** - Schedule periodic satisfaction reviews",
                "📱 **Engagement programs** - Invite to customer community or events"
            ]
            
            for rec in recommendations:
                st.write(rec)
    
    def create_sample_data(self) -> pd.DataFrame:
        """
        Crea datos de muestra para demostración.
        """
        np.random.seed(42)
        
        n_samples = 50
        
        data = {
            'customer_id': [f"CUST-{i:04d}" for i in range(1, n_samples + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'senior_citizen': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
            'partner': np.random.choice([True, False], n_samples, p=[0.5, 0.5]),
            'dependents': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'phone_service': np.random.choice([True, False], n_samples, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['No', 'Yes', 'No phone service'], n_samples, p=[0.4, 0.3, 0.3]),
            'internet_service': np.random.choice(['No', 'DSL', 'Fiber optic'], n_samples, p=[0.1, 0.4, 0.5]),
            'online_security': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.5, 0.3, 0.2]),
            'online_backup': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'device_protection': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.5, 0.3, 0.2]),
            'tech_support': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.6, 0.2, 0.2]),
            'streaming_tv': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'streaming_movies': np.random.choice(['No', 'Yes', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.2, 0.2]),
            'paperless_billing': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
            'payment_method': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples, p=[0.4, 0.2, 0.2, 0.2]),
            'monthly_charges': np.round(np.random.uniform(20, 120, n_samples), 2),
            'tenure_months': np.random.randint(1, 72, n_samples)
        }
        
        # Calcular total_charges basado en tenure y monthly_charges
        data['total_charges'] = np.round(data['tenure_months'] * data['monthly_charges'], 2)
        
        return pd.DataFrame(data)
    
    def row_to_customer_dict(self, row: pd.Series) -> Dict[str, Any]:
        """
        Convierte una fila de DataFrame a diccionario de cliente.
        """
        return {
            'customer_id': str(row.get('customer_id', 'UNKNOWN')),
            'gender': str(row.get('gender', 'Female')),
            'senior_citizen': bool(row.get('senior_citizen', False)),
            'partner': bool(row.get('partner', False)),
            'dependents': bool(row.get('dependents', False)),
            'phone_service': bool(row.get('phone_service', False)),
            'multiple_lines': str(row.get('multiple_lines', 'No')),
            'internet_service': str(row.get('internet_service', 'No')),
            'online_security': str(row.get('online_security', 'No')),
            'online_backup': str(row.get('online_backup', 'No')),
            'device_protection': str(row.get('device_protection', 'No')),
            'tech_support': str(row.get('tech_support', 'No')),
            'streaming_tv': str(row.get('streaming_tv', 'No')),
            'streaming_movies': str(row.get('streaming_movies', 'No')),
            'contract_type': str(row.get('contract_type', 'Month-to-month')),
            'paperless_billing': bool(row.get('paperless_billing', False)),
            'payment_method': str(row.get('payment_method', 'Electronic check')),
            'monthly_charges': float(row.get('monthly_charges', 0.0)),
            'total_charges': float(row.get('total_charges', 0.0)),
            'tenure_months': int(row.get('tenure_months', 0))
        }
    
    def color_risk_level(self, row: pd.Series) -> List[str]:
        """
        Aplica color a las filas basado en el nivel de riesgo.
        """
        risk_colors = {
            'Low': 'color: #2ca02c',
            'Medium': 'color: #ff7f0e',
            'High': 'color: #d62728',
            'Critical': 'color: #8c564b; font-weight: bold'
        }
        
        return [risk_colors.get(row['risk_level'], '')] * len(row)
    
    def show_detailed_statistics(self, results_df: pd.DataFrame):
        """
        Muestra estadísticas detalladas de los resultados.
        """
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Statistics:**")
            pred_stats = results_df['prediction'].value_counts()
            st.write(f"- Churn: {pred_stats.get(True, 0)} customers")
            st.write(f"- No Churn: {pred_stats.get(False, 0)} customers")
            
            st.write("**Risk Level Statistics:**")
            risk_stats = results_df['risk_level'].value_counts()
            for level, count in risk_stats.items():
                percentage = (count / len(results_df)) * 100
                st.write(f"- {level}: {count} customers ({percentage:.1f}%)")
        
        with col2:
            st.write("**Probability Statistics:**")
            prob_stats = results_df['probability_churn'].describe()
            st.write(f"- Mean: {prob_stats['mean']:.3f}")
            st.write(f"- Median: {prob_stats['50%']:.3f}")
            st.write(f"- Std Dev: {prob_stats['std']:.3f}")
            st.write(f"- Min: {prob_stats['min']:.3f}")
            st.write(f"- Max: {prob_stats['max']:.3f}")


def main():
    """
    Función principal de la aplicación Streamlit.
    """
    app = ChurnPredictionApp()
    app.run()


if __name__ == "__main__":
    main()
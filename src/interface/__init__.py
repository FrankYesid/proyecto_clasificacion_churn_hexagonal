"""
Módulo de inicialización para el paquete de interfaces.
Exporta las interfaces principales disponibles.
"""

# FastAPI
from .api.main import app

# Streamlit (se cargará dinámicamente)
try:
    import streamlit
    from .web.streamlit_app import main as streamlit_main
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    streamlit_main = None

__all__ = [
    'app',  # FastAPI app
    'streamlit_main',  # Streamlit main function
    'STREAMLIT_AVAILABLE'
]
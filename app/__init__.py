"""
FastAPI web service for the IPO Listing Gain Prediction model.

This package serves the trained pipeline behind an HTTP ``/predict`` endpoint,
handling request validation, schema-to-DataFrame mapping, and prediction
logging. It is the production inference surface consumed by the Streamlit
dashboard and any external clients.

Run:
    python -m app.main

Override the bind address via env vars (``HOST``, ``PORT``) if needed.
"""

import pytest
from app import app


@pytest.fixture
def client():
    return app.test_client()


def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Hello from Flask" in response.data



def test_predict(client):
    test_data = {
        'Gender': 1,              # Male → 1
        'Married': 0,             # Unmarried → 0
        'Credit_History': 0,      # Unclear Debts → 0
        'Total_Income_Log': 10.5  # Some numeric log value
    }

    resp = client.post('/predict', json=test_data)
    assert resp.status_code == 200
    assert resp.json == {'prediction': 'Loan Rejected'}
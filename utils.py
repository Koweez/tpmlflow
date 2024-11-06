import mlflow

def load_model(model_name):
    # Load the model
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    return model

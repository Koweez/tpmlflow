import fastapi
from utils import load_model

app = fastapi.FastAPI()

model = load_model("tracking-quickstart", "version1")
next_model = load_model("tracking-quickstart", "version2")
canary_proba = 0.1
actual_proba = 0.0

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    if actual_proba < canary_proba:
        model = next_model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return {"prediction": prediction[0].item()}

@app.get("/update-model")
def update_model():
    global model
    model = load_model("tracking-quickstart")
    return {"message": "Model updated successfully"}

@app.get("/accept-next-model")
def update_probability():
    global actual_proba
    actual_proba = next_model
    return {"message": "Model accepted successfully"}

import fastapi
from utils import load_model

app = fastapi.FastAPI()

model = load_model("tracking-quickstart")

@app.get("/")
def read_root():
    return {"Hello": "World"}

# predict user input
@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return {"prediction": prediction[0].item()}

@app.get("/update-model")
def update_model():
    global model
    model = load_model("tracking-quickstart")
    return {"message": "Model updated successfully"}

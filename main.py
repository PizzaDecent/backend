from typing import Union
from internal import scan
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/predict")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(contents)

    return scan.predict_damage(f"uploads/{file.filename}")



@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
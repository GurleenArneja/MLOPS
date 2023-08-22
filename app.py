from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import RedirectResponse
from model import Model
import logging
# from typing import Any
import uvicorn

logging.basicConfig(level = logging.INFO)
API_V1_STR: str = "/api/v2"

app = FastAPI(
    title = "Sentiment Analaysis", openapi_url = f"{API_V1_STR}/openapi.json"
)
router = APIRouter()
model = Model()

@router.get("/")
async def home(request: Request):
    return RedirectResponse(url="/docs")

@router.post("/sentiment")
async def data(data: dict):
    try:
        input_text = data["text"]
        res = model.sentimentAnalysis(input_text, API_V1_STR)
        return res
    except Exception as e:
        logging.error("Something went wrong", e)

app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True, host="localhost", port=8001)

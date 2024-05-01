
import requests
from models import potential_growth

from fastapi import FastAPI

app = FastAPI()



@app.get("/potential_growth")
async def potential_growth_get():
    """
    Echoes the received text back in the response.
    """
    return {"accuracy": potential_growth()}
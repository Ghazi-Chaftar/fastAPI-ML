
import requests
from models import potential_growth

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration du middleware CORS
origins = [
    "http://localhost",
    "http://localhost:4200",  # Ajoutez l'URL de votre application Angular ici
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.post("/potential_growth")
async def potential_growth_get(request:dict):
    """
    Echoes the received text back in the response.
    """
    return {"accuracy": potential_growth(request)}
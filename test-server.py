import requests

print( 
    requests.get(
        "http://localhost:8000/potential_growth",
    ).json()
)
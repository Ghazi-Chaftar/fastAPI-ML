import requests

print( 
    requests.post(
        "http://localhost:8000/potential_growth",
        json={
            'TYPE_DIPLOMA': 2,
            'EXP_YEARS': 1,
            'POSITION': 0,
            'SOURCE_of_employment': 0,
            'TYPE_CONTRACT': 2
        }
    ).json()
)
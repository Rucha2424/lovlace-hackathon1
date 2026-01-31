from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analysis import build_cell_signal, infer_topology, compute_capacity

app = FastAPI()

# 🔴 CORS FIX (THIS IS THE KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5500',
        'http://127.0.0.1:5500'
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

CACHE = None

@app.get('/analyze')
def analyze():
    global CACHE

    if CACHE is None:
        cell_signal = build_cell_signal()
        topology = infer_topology(cell_signal)
        capacity = compute_capacity(topology)

        CACHE = {
            'topology': topology,
            'capacity': capacity
        }

    return CACHE

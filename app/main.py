from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import __version__ as ml_model_version 
from app.model.model import get_recommender
from typing import List


app = FastAPI()

# # string input
# class Input_String (BaseModel):
#     text: str

# # output_list
class Final_Recomndation(BaseModel): 
    recommendation : tuple

# # input integer
# class Input_Integer (BaseModel):
#     integer: int

# # input input
# class Input_List (BaseModel):
#     integer: int 
# # starting the url
# # base url
class InputPayload(BaseModel):
    bahan_yang_disukai: List[str]
    bahan_yang_tidak_disukai: List[str]
    pantangan_makan: List[str]
    budget: int
    jumlah_makan_sehari: int
    jumlah_dewasa: int
    jumlah_anak: int
    country_references: List[str]
    


@app.get("/")
def home():
    return {"model_version": ml_model_version}


# get the recomendation function 
#  parameter to get the recomender : bahan_yang_disukai = ['Nasi', 'Bayam', 'Sapi', 'Babi'], bahan_yang_tidak_disukai = 'Keju' , pantangan_makan = '', budget = 2000000, jumlah_makan_sehari = 3, jumlah_dewasa = 2, jumlah_anak  = 2

@app.post("/predict", response_model=Final_Recomndation)
def predict(payload: InputPayload):
    # /bahan_yang_disukai, bahan_yang_tidak_disukai , pantangan_makan, budget, jumlah_makan_sehari, jumlah_dewasa, jumlah_anak, country_refrences_param
    recommendation = get_recommender(payload.bahan_yang_disukai, payload.bahan_yang_tidak_disukai, payload.pantangan_makan, payload.budget, payload.jumlah_makan_sehari, payload.jumlah_dewasa, payload.jumlah_anak, payload.country_references)
    return {'recommendation': recommendation}

from pydantic import BaseModel, Field
from typing import Literal

class TitanicPassenger(BaseModel):
    prefix: Literal['Mr','Miss','Mrs','Master','Mlle','Rev'] = Field(example='Mr')
    first_name: str = Field(example=['James','Michael'])
    family_name: str = Field(example=['Myles','Kelly']) 
    gender: Literal['male','female'] = Field(example='male')
    age: int = Field(ge=1, le=80, example=[5, 76])
    passenger_id: int = Field(ge=1, le=891, example=[43, 678])
    number_siblings: int = Field(ge=0, le=8, example=[2,6])
    number_parch: int = Field(ge=0, le=6, example=[0,2])
    ticket_name: str = Field(example=['330911','PC77658','SOTON787873'])
    cabin_name: str = Field(example=['A40','C10'])
    pclass: int = Field(ge=1, le=3, example=2)
    fare_price: float = Field(ge=0, le=512, example=407.2)
    embarked_port: Literal['S','C','Q'] = Field(example='S')



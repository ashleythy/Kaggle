from pydantic import BaseModel

# 

class TitanicPassenger(BaseModel):
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    len_unq_firstname: int
    len_char_firstname: int


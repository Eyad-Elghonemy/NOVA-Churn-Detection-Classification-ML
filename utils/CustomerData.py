from pydantic import BaseModel, Field
from typing import Literal

class CustomerData(BaseModel):

        CreditScore: int = Field(description='Credit Score Of The Customer')
        Geography: Literal['France', 'Spain', 'Germany'] = Field(description="Customer's Country")
        Gender: Literal['Male', 'Female'] = Field(description="Customer's Gender")
        Age: int = Field(description="Customer's Age", ge=18, le=100)
        Tenure: int = Field(description="Years As A Customer (0-10)", ge=0, le=10)
        Balance: float = Field(description="Account Balance", ge=0)
        NumOfProducts: int = Field(description="Number Of Bank Products (1-4)", ge=1, le=4)
        HasCrCard: Literal[0, 1] = Field(description="Has Credit Card (0:No, 1:Yes)")
        IsActiveMember: Literal[0, 1] = Field(description="Active Member Status (0:No, 1:Yes)")
        EstimatedSalary: float = Field(description="Estimated Annual Salary", ge=0)
    
    
    
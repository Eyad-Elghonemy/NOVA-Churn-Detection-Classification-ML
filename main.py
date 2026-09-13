from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from utils.inference import predict_new
from utils.config import APP_NAME, VERSION, SECRET_KEY_TOKEN, preprocessor, forest_model, xgboost_model
from utils.CustomerData import CustomerData
from utils.usage_tracker import log_usage, get_stats

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
        CORSMiddleware, 
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
)



api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str=Depends(api_key_header)):
        if api_key != SECRET_KEY_TOKEN:
                raise HTTPException(status_code=403, detail="You Are Not Authorized To Use This API")
        
        return api_key



@app.get('/', tags=['General'])  
async def home():
    return {
            "Message": f"Welcome To My {APP_NAME} API v{VERSION}"
    }



@app.get('/stats', tags=['General'])
async def stats(api_key: str = Depends(verify_api_key)) -> dict:
        return get_stats()
    
    

@app.post('/predict/forest', tags=['Models'])  
async def predict_forest(data: CustomerData, api_key: str=Depends(verify_api_key)) -> dict:
        
        try:
                result = predict_new(data=data, preprocessor=preprocessor, model=forest_model)
                log_usage("forest")
                return result
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.post('/predict/xgboost', tags=['Models'])  
async def predict_xgboost(data: CustomerData, api_key: str=Depends(verify_api_key)) -> dict:
        
        try:
                result = predict_new(data=data, preprocessor=preprocessor, model=xgboost_model)
                log_usage("xgboost")
                return result
        
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI

app = FastAPI(title="Scalp Vision Agent API")

@app.get("/health")
def health_check() :
    return {"message" : "scalp-vision-agent api ok"}
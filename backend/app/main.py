from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vector_store import vector_store

# from model import load_model, encode_query

app = FastAPI()


# Load Model
class QueryRequest(BaseModel):
    text: str
    top_k: int = 5


@app.post("/search")
async def search_documents(request: QueryRequest):
    results = vector_store.similarity_search(
        request.text, k=request.top_k, filter={"source": "search"}
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    return {"documents": [res.page_content for res in results]}

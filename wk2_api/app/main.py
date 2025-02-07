import os
from fastapi import FastAPI
import numpy as np
import json 
import gensim.downloader as api
from app.chroma_embed import Embeddings
from langchain_chroma import Chroma


# Write the model version here or find some way to derive it from the model
# eg. from the model files name
model_path = "/Users/yuliagoryachev/Desktop/mlx_word2vec_ver_0.1.0"
model_version = model_path.split("_")[-1]

# Set the log path.
# This should be a directory that is writable by the application.
# In a docker container, you can use /var/log/ as the directory.
# Mount this directory to a volume on the host machine to persist the logs.
log_dir_path = "/var/log/app"
log_dir_path_local = "./logs"
log_path = f"{log_dir_path_local}/V-{model_version}.log"

app = FastAPI()

# Define the endpoints
#
#
#
@app.get("/ping")
def ping():
  return "ok"


@app.get("/version")
def version():
  return {"version": model_version}


@app.get("/logs")
def logs():
  return read_logs(log_path)


@app.post("/k_nearest_documents")
def k_nearest_documents(search_query: str, k: int):
    post = {"search": search_query, "k": k}
    start_time = os.times().elapsed     # Start time for latency calculation
    prediction = extract_documents(post)  # Placeholder for actual prediction
    end_time = os.times().elapsed       # End time for latency calculation
    latency = (end_time - start_time) * 1000

    message = {}
    message["Latency"]    = latency
    message["Version"]    = model_version
    message["Timestamp"]  = end_time
    message["Input"]      = json.dumps(post)
    message["Prediction"] = prediction

    print(f"Message: {message}\n")
    log_request(log_path, json.dumps(message))
    return {"upvotes": prediction}



# Functions for logging and for predicting upvotes
##### Placeholder for actual prediction #####
def extract_documents(post):
    #post should have the following keys: "search", "k"

    query = post["search"]
    k = post["k"]
    persist_directory = "../chroma_langchain_db"  # adjust this path if necessary
    collection_name = "mlx_wk2_collection"

    emb = Embeddings()

    # Create (or load) the Chroma vector store.
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=emb,
        collection_name=collection_name  # this parameter is optional
    )

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"source": "search"},
    )
    print('Question: ', query)

    #convert results to str
    res = [str(res.page_content) for res in results]
    return res


##### Log The Request #####
def log_request(log_path, message):
  # print the message and then write it to the log
  pass


##### Read The Logs #####
def read_logs(log_path):
  # read the logs from the log_path
  pass
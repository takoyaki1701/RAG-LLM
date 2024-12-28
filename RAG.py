from get_Sentence import querySentence
from chunk_Split import GetChunk
from sentence_Embedding import Embedding
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def RAG(file_path, query):
    # sentences into list
    text = querySentence(file_path)
    list_of_sentence = text.extractSentence()
    # Split docs into chunks
    chunks = GetChunk(list_of_sentence)
    chunk = chunks.splitChunks()

    # embed chunks 
    embedded = Embedding(chunk)
    embeddedChunk, pages_and_chunks = embedded.EmbedModel()
    embeddedChunk = embeddedChunk.to(device)
    print("Embedded chunks: \n", embeddedChunk)

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.to(device)
    print("Embedded query: \n", query_embedding)

    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddedChunk)[0]
    end_time = timer()

    rag_output = ""
    assistant_messages = """ If you don't know the answer, use this information to answer the question:\n
"""

    rag_output += f"Time take to get scores on {len(embeddedChunk)} embeddings: {end_time-start_time:.5f} seconds. \n"

    # 4. Get the top-k results (3)
    top_results_dot_product = torch.topk(dot_scores, k=3)

    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        rag_output += f"Score: {score:.4f}\n"
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        rag_output += "Text: \n"
        rag_output += f"""{pages_and_chunks[idx]["sentence_chunk"]} \n"""
        # Print the page number too so we can reference the textbook further (and check the results)
        rag_output += f"Page number: {pages_and_chunks[idx]['page_number']} \n"
        assistant_messages += pages_and_chunks[idx]["sentence_chunk"]
    
    return assistant_messages, rag_output
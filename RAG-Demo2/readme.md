This is a small teaching/demo project for a simple Retrieval-Augmented Generation (RAG) pipeline and a lightweight evaluation layer.

High-level idea

rag.py builds a minimal RAG app:
Reads a knowledge source
Splits it into chunks
Converts chunks and the user question into embeddings
Finds the most relevant chunk by cosine similarity
Sends that chunk + the question to an Azure OpenAI chat model
metrics.py defines four “RAGAS-like” evaluation metrics using embedding similarity as a proxy
test_metrics.py exercises the full pipeline and checks that the scores are above a loose threshold
knowledge.txt is the knowledge base the app retrieves from
What each file does

rag.py

get_client(): creates the Azure OpenAI client using environment variables
load_chunks(): loads the text file and splits it on blank lines into paragraph-sized chunks
embed_text(): sends text to the embeddings model and returns the vector
cosine_similarity(): computes similarity between two embedding vectors
retrieve_top_chunk(): embeds the query and picks the chunk with the highest cosine similarity
answer_question(): does the real end-to-end workflow:
load chunks
embed all chunks
retrieve the best chunk
call the chat model with “context + question”
return (retrieved_context, answer)
metrics.py

These are simplified versions of RAG evaluation metrics:
faithfulness_score(answer, context): answer vs retrieved context
answer_relevancy_score(answer, question): answer vs user question
context_precision_score(context, question): context vs question
context_recall_score(context, reference_answer): context vs known-good answer
The code does not do true claim decomposition or LLM-as-judge checks; it just embeds both texts and measures similarity.
test_metrics.py

Uses a real prompt/question (What is RAG?)
Runs the pipeline once via a shared fixture
Calls each metric and asserts score > 0.5
Prints the numeric score for each metric
How the app works in practice

The app expects Azure OpenAI environment variables:
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBEDDING_DEPLOYMENT
AZURE_OPENAI_CHAT_DEPLOYMENT
It uses the Azure AI Foundry /openai/v1 endpoint pattern.
The retrieval step is very simple: vector similarity search over the knowledge file.
The answer generation step uses only the best retrieved chunk as context, with the instruction: “Answer using only the provided context.”
Important caveat

This is intentionally simplified for teaching. The comments in metrics.py explicitly say:

it is not production-grade RAGAS
it reduces metrics to embedding similarity
real RAGAS does more advanced claim-level and LLM-based evaluation
So this codebase is best understood as:

a minimal Azure OpenAI-based RAG demo
a hands-on example of embedding-based retrieval and evaluation
a classroom version of RAG metrics, not a full production evaluator


Here is the exact runtime path for a single user query in this project, using the CLI entry point in rag.py.

The script starts in __main__
At the bottom of rag.py, this runs:
question = " ".join(sys.argv[1:]) or "What is RAG?"
then context, answer = answer_question(question)
So the user input becomes the query string. If you run:
python rag.py "What is RAG?"
the query is exactly "What is RAG?"
answer_question(query) is called
That function is defined in rag.py.
It does the whole RAG flow in one place.
It creates the Azure OpenAI client
client = get_client()
get_client() reads:
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
Then it builds:
OpenAI(base_url=..., api_key=...)
The base URL is expected to already include the Azure AI Foundry /openai/v1 suffix.
It reads the required deployment names
embed_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
It loads the knowledge base
chunks = load_chunks()
load_chunks() reads knowledge.txt
It reads the whole file as text, then splits it on double newlines:
text.split("\n\n")
It trims whitespace and throws away empty chunks
Result: a list like ["paragraph 1", "paragraph 2", ...]
It embeds every chunk
For each chunk:
embed_text(client, chunk, embed_deployment)
That calls:
client.embeddings.create(input=text, model=deployment)
Azure returns an embedding vector
So now you have:
chunk_embeddings = [vector_for_chunk1, vector_for_chunk2, ...]
It retrieves the best matching chunk
retrieve_top_chunk(query, chunks, chunk_embeddings, client, embed_deployment)
Inside that:
query_embedding = embed_text(client, query, embed_deployment)
then for each chunk vector:
cosine_similarity(query_embedding, chunk_embedding)
This computes similarity between the user question and each knowledge chunk
The chunk with the highest score is chosen:
best_index = scores.index(max(scores))
return chunks[best_index]
That selected chunk becomes the retrieved context
The function returns:
context = "the single most relevant paragraph from knowledge.txt"
It asks the chat model to answer using only that context
The code then does:
response = client.responses.create(...)
With:
model=chat_deployment
instructions="Answer the question using only the provided context. If the context doesn't contain the answer, say you don't know."
input=f"Context:\n{context}\n\nQuestion: {query}"
This is the key prompt:

It tells the model:
“Use this retrieved text only”
“Do not answer from your general knowledge”
“If not in the context, say you don’t know”
The model returns the answer
response.output_text
That is the generated answer text
The function returns:
(retrieved_context, answer)
The script prints both pieces
In __main__:
print(f"Context used:\n{context}\n")
print(f"Answer:\n{answer}")
So the end-to-end flow is:

user query
build Azure client
load knowledge file
split into chunks
embed all chunks
embed the query
compute cosine similarity between question and each chunk
select highest-scoring chunk
send chunk + question to chat model
model answers using only that chunk
print context and answer
How evaluation fits into this

The evaluation code in metrics.py uses the same idea after the answer is created:
compare answer vs context
compare answer vs question
compare context vs question
compare context vs a reference answer
It does this by embedding the text pairs and computing cosine similarity, which is a lightweight proxy for RAG evaluation.
One subtle but important point:

The project is intentionally simplified. It is not doing deep retrieval, chunk reranking, or true claim-level evaluation. It is a teaching demo of “retrieval + similarity-based scoring.”
If you want, I can also walk through the same flow using the exact values from a sample run, like:

the raw knowledge.txt chunk chosen for “What is RAG?”
the exact embedding similarity math
what the final prompt sent to the model looks like.
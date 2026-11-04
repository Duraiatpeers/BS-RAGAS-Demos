What this project does
This is a small Retrieval-Augmented Generation (RAG) demo. It takes a question, finds the most relevant paragraph in a local knowledge file, and asks an Azure OpenAI chat model to answer using that paragraph.

The main files are rag.py, which implements the pipeline; knowledge.txt, which provides its source material; and test_rag.py, which checks retrieval. The two flow diagrams, rag_pipeline_flow.png and test_rag_flow.png, visualize the app and test paths.

How a question is answered
Load the knowledge: load_chunks() reads knowledge.txt and splits it into paragraphs.
Prepare the Azure client: get_client() configures the OpenAI SDK using the Azure endpoint and API key.
Create embeddings: embed_text() sends each paragraph—and later the question—to the configured embedding deployment. An embedding is a numeric representation of text meaning.
Find relevant context: retrieve_top_chunk() compares the question’s embedding to each paragraph’s embedding using cosine similarity, then returns the closest paragraph.
Generate the answer: answer_question() sends that paragraph and the question to the chat deployment. Its instructions tell the model to answer from the supplied context or say it doesn’t know.
Print the result: Running rag.py prints the answer. If no question is supplied, it asks “What is RAG?”
Configuration and running
rag.py loads a local .env file, if present. It expects these environment variables:

AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBEDDING_DEPLOYMENT
AZURE_OPENAI_CHAT_DEPLOYMENT
Run it with python rag.py "What is RAG?" after setting those values. The test can be run with pytest test_rag.py -v.

What the test checks
The test makes real Azure embedding API calls for the knowledge paragraphs and a question about cosine similarity. It then asserts that the retrieved paragraph contains “Cosine similarity.” It tests retrieval—not the final chat-model answer—and requires working Azure credentials and deployments.


assert "Cosine similarity" in result is a check in the test:

result is the paragraph returned by retrieve_top_chunk().
"Cosine similarity" in result checks whether that exact phrase appears in the paragraph.
assert makes the test pass if it does, and fail if it doesn’t.
So the test is checking that the retrieval step picked the paragraph about cosine similarity. It’s case-sensitive and checks for the phrase anywhere in the paragraph—not that the whole paragraph matches.


A couple of limitations
This is a deliberately minimal demo: it retrieves only one paragraph, and it recomputes embeddings for every question, which adds latency and API usage. It also assumes the knowledge file contains at least one paragraph.
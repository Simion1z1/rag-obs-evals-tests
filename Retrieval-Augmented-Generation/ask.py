import uuid

import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Langfuse must be imported after load_dotenv() so it picks up the right credentials,
# and the Groq instrumentor must be set up before any Groq client is used.
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.groq import GroqInstrumentor

GroqInstrumentor().instrument()

langfuse = get_client()
assert langfuse.auth_check(), "Langfuse auth failed - check your LANGFUSE_* env vars"

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")

client = Groq()

# One session per CLI run so all questions in this conversation are grouped together in Langfuse.
session_id = str(uuid.uuid4())

print("Ask me anything about growing vegetables in Florida. Type 'exit' or 'quit' to stop.\n")

while True:
    user_query = input("What do you want to know about growing vegetables?\n\n")

    if user_query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not user_query.strip():
        continue

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-query",
        input=user_query,
    ) as root_span, propagate_attributes(
        session_id=session_id,
        tags=["rag", "vegetables-qa"],
    ):
        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="chroma-retrieval",
            input=user_query,
        ) as retrieval_span:
            results = collection.query(
                query_texts=[user_query],
                n_results=4
            )
            retrieval_span.update(output=results['documents'])

        print(results['documents'])
        #print(results['metadatas'])

        system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida.
But you only answer based on knowledge I'm providing you. You don't use your internal
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""

        #print(system_prompt)

        # The Groq call below is auto-instrumented (model, prompts, tokens, latency)
        # by GroqInstrumentor and nested under this span as a "generation" observation.
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_query}
            ]
        )

        answer = response.choices[0].message.content
        root_span.update(output=answer)

    langfuse.flush()

    print("\n\n---------------------\n\n")

    print(answer)

    print("\n=====================\n")

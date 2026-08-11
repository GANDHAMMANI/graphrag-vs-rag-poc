# test_answer_model.py, project root
from core.retrieve import LlamaIndexGraphRetriever

retriever = LlamaIndexGraphRetriever()

for q in [
    "Who is the CEO of Rodriguez, Figueroa and Sanchez?",
    "What did Courtney Keller discuss in her meeting with Carl Gentry?",
    "What products were discussed in Patricia Marshall's meeting with Tracie Wyatt?",
]:
    result = retriever.retrieve_with_citations(q)
    print(f"\nQ: {q}\nA: {result['answer']}\n")
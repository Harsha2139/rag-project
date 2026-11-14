
from src.retriever import build_search_system, retrieve
from src.llm_generator import generate_answer

print("Building your RAG search system...")
store, metadata = build_search_system()

print("\nRAG Application is ready!")

while True:

    question = input("\nAsk a question (type exit to stop): ")

    if question.lower() == "exit":
        print("Thank you for using the RAG application!")
        break

    results = retrieve(question, store, metadata)

    if not results:
        print("No relevant document content found.")
        continue

    context = "\n\n".join(
        f"Page {item['page']}: {item['text']}"
        for item in results
    )

    print("\nRetrieved document context:")
    print(context)

    answer = generate_answer(question, context)

    print("\nAI Answer:")
    print(answer)

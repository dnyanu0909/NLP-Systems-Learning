def generate_response(query, retrieved_docs):
    context = [doc for doc, _ in retrieved_docs]

    return f"""
I understand you're dealing with: "{query}"

Here are some practical steps:

- {context[0]}
- {context[1]}
"""
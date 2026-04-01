def rewrite_query(query, history):
    """
    Makes query more context-aware
    """
    if len(history) == 0:
        return query

    last_context = " ".join(history[-2:])

    return f"{last_context} {query}"
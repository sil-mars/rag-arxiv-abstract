class ContextB:

    def build(self, docs, k=3):

        # Take the best ones
        top_docs = docs[:k]
        
        # Extract text from each document
        context = "\n\n".join([d["text"] for d in top_docs])

        return context

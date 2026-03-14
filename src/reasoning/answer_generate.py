
# src/reasoning/answer_generator.py

class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question, retrieved_chunks):
        if len(retrieved_chunks) == 0:
            return "I don't know yet."

        context = "\n".join(f"- {c}" for c in retrieved_chunks)

        prompt = f"""
Use ONLY the information below to answer.
If the information is insufficient, say "I don't know yet."

Knowledge:
{context}

Question:
{question}

Answer:
"""

        return self.llm.generate(prompt)

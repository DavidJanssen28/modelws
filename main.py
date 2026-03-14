from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.reasoning.llm_interface import LLMInterface
from src.reasoning.answer_generator import AnswerGenerator
from src.interfaces.teach_interface import Teacher

embedder = Embedder()
vector_store = VectorStore()
retriever = Retriever(vector_store, embedder)
llm = LLMInterface()
answer_gen = AnswerGenerator(llm)
teacher = Teacher(embedder, vector_store)

def ask(question):
    indices, _ = retriever.retrieve(question)
    chunks = []

    for idx in indices:
        try:
            with open(f"knowledge/raw/entry_{idx+1:03d}.txt") as f:
                chunks.append(f.read())
        except:
            pass

    return answer_gen.generate(question, chunks)

def teach(text):
    return teacher.teach(text)


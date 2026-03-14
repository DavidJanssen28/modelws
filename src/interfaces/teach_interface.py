# src/interfaces/teach_interface.py

import os
from datetime import datetime

class Teacher:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.raw_dir = "knowledge/raw/"

    def teach(self, text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.raw_dir}/entry_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(text)

        vec = self.embedder.embed(text)
        self.vector_store.add(vec)

        return filename


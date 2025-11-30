import os
import torch
import pandas as pd
import transformers
from pynvml import *
import torch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from model_ret import load_model_and_pipeline
from create_retriever import retriever_chroma

# Model chain class
class model_chain:
    model_name = ""

    def __init__(self,
                 model_name_local,
                 model_name_online="Llama",
                 use_online=True,
                 embedding_name="BAAI/bge-base-en-v1.5",
                 splitter_type_dropdown="character",
                 chunk_size_slider=512,
                 chunk_overlap_slider=30,
                 separator_textbox="\n",
                 max_tokens_slider=2048) -> None:
        if os.path.exists(f"models//{model_name_local}") and len(os.listdir(f"models//{model_name_local}")):
            import gradio as gr
            gr.Info("Model *()* from online!!")
            self.model_name = model_name_local
        else:
            self.model_name = model_name_online

        self.tokenizer, self.model, self.llm = load_model_and_pipeline(self.model_name)
        # Creating the retriever
        # self.retriever = ensemble_retriever(embedding_name,
        #                                     splitter_type=splitter_type_dropdown,
        #                                     chunk_size=chunk_size_slider,
        #                                     chunk_overlap=chunk_overlap_slider,
        #                                     separator=separator_textbox,
        #                                     max_tokens=max_tokens_slider)
        self.retriever = retriever_chroma(False, embedding_name, splitter_type_dropdown,
                                          chunk_size_slider, chunk_size_slider,
                                          separator_textbox, max_tokens_slider)

        # Defining the RAG chain
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # Helper function to format documents
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Retrieve RAG chain
    def rag_chain_ret(self):
        return self.rag_chain

    # Answer retrieval function
    def ans_ret(self, inp):
        if self.model_name == 'Flant5':
            my_question = "What is KUET?"
            data = self.retriever.invoke(inp)
            context = ""
            for x in data[:2]:
                context += (x.page_content) + "\n"
            inputs = f"""Please answer to this question using this context:\n{context}\n{my_question}"""
            inputs = self.tokenizer(inputs, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            answer = self.tokenizer.decode(outputs[0])
            from textwrap import fill
            ans = fill(answer, width=100)
            return ans

        ans = self.rag_chain.invoke(inp)
        ans = ans.split("Answer:")[1]
        return ans
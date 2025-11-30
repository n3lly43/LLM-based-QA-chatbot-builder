import os
import torch
from time import time
from torch import cuda, bfloat16
from langchain import HuggingFacePipeline
from transformers import (AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, AutoConfig,
                          T5ForConditionalGeneration, pipeline, BitsAndBytesConfig)

def load_model_and_pipeline(model_id, temperature=0):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    time_1 = time()
    model_config = AutoConfig.from_pretrained(
        model_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    time_2 = time()
    print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")
    time_1 = time()

    pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=torch.float16,
            device_map="auto",
            max_new_tokens=512,
            do_sample=True,
            top_k=30,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
    time_2 = time()
    print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': temperature})
    return tokenizer, model, llm

import pandas as pd
from datasets import Dataset

def calculate_rag_metrics(model_ques_ans_gen, llm_model, embedding_model="BAAI/bge-base-en-v1.5"):
    # Create a dictionary from the model_ques_ans_gen list
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_correctness,answer_similarity,answer_relevancy,context_recall, context_precision
    # context = {'contexts':'NA'}
    # print([item['question'] for item in model_ques_ans_gen])
    data_samples = {
        'question': [item['question'] for item in model_ques_ans_gen],
        'answer': [item['answer'] for item in model_ques_ans_gen],
        'contexts': [[''] for item in model_ques_ans_gen],
        'reference': [item['ground_truths'] for item in model_ques_ans_gen]
    }

    # Convert the dictionary to a pandas DataFrame
    rag_df = pd.DataFrame(data_samples)

    # Convert the DataFrame to a HuggingFace Dataset
    rag_eval_dataset = Dataset.from_pandas(rag_df)

    # Define the list of metrics to calculate
    metrics = [
        answer_correctness, answer_similarity,
        answer_relevancy, faithfulness,
        context_recall, context_precision
    ]

    # Perform the evaluation using the provided LLM and embedding models
    result = evaluate(
        rag_eval_dataset,
        metrics=metrics,
        llm=llm_model,
        embeddings=embedding_model
    )
    # result.to_pandas()
    return result.to_pandas()
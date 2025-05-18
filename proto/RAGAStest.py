import GraphRAG
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas.metrics import ContextRecall, LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference
from ragas.metrics import  SemanticSimilarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
import json
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
wrapper_llm = LangchainLLMWrapper(llm)
wrapper_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-large"))


def create_dataset(path):
    
    loader = DirectoryLoader(path, glob="**/[!.]*")
    docs = loader.load()
    
    generator = TestsetGenerator(wrapper_llm, wrapper_emb)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
    dataset.upload()
    return dataset

def run_tests(dataTest):
    dataset = []

    for entry in dataTest:
        query = entry["question"]
        reference = entry["answer"]

        # Recupera i chunk di contesto rilevanti e la risposta
        retrieved_context = GraphRAG.return_context(query)
        response = GraphRAG.ask_data(query)
        
        dataset.append({
            "user_input": query,
            "retrieved_contexts": retrieved_context,
            "response": response,
            "reference": reference
        })
        

    eval_dataset = EvaluationDataset.from_list(dataset)
    result = evaluate(dataset=eval_dataset, metrics=[LLMContextPrecisionWithReference(), ContextRecall(), ResponseRelevancy(), Faithfulness(), SemanticSimilarity(embeddings = wrapper_emb)], llm=wrapper_llm)

    result.upload()
    
    return result


def run_tests_RAG(dataTest):
    dataset = []

    for entry in dataTest:
        query = entry["question"]
        reference = entry["answer"]

        # Recupera i chunk di contesto rilevanti e la risposta
        retrieved_context = GraphRAG.return_node(query)
        response = GraphRAG.ask_data_RAG(query)
        
        dataset.append({
            "user_input": query,
            "retrieved_contexts": retrieved_context,
            "response": response,
            "reference": reference
        })
        

    eval_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm) 
    result = evaluate(dataset=eval_dataset, metrics=[LLMContextPrecisionWithReference(), ContextRecall(), ResponseRelevancy(), Faithfulness(), SemanticSimilarity(embeddings = wrapper_emb)], llm=evaluator_llm)


    result.upload()
    
    return result



if __name__ == "__main__":
    with open("../tests/data/qa/Liliana_qa.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    result_graph = run_tests(dataset)
    result_rag = run_tests_RAG(dataset)
    print(result_graph)
    print(result_rag)
    
    
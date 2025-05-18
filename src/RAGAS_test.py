from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    ContextRecall,
    ResponseRelevancy,
    Faithfulness,
    SemanticSimilarity,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

import GraphRAG  

# Load environment variables
load_dotenv()

# Model and Wrapper Initialization
llm = ChatOpenAI(model="gpt-4o-mini")
llm_wrapper = LangchainLLMWrapper(llm)
embedding_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-large"))


def create_evaluation_dataset(path: str):
    """
    Create and upload a test dataset from documents in a directory.

    Args:
        path: Directory path with QA documents.

    Returns:
        Uploaded EvaluationDataset object.
    """
    loader = DirectoryLoader(path, glob="**/[!.]*")
    documents = loader.load()

    generator = TestsetGenerator(llm_wrapper, embedding_wrapper)
    dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
    dataset.upload()

    return dataset


def evaluate_graphRAG(data_test: list):
    """
    Evaluate the performance of the GraphRAG approach.

    Args:
        data_test: List of dictionaries with 'question' and 'answer'.

    Returns:
        EvaluationResult object from RAGAS.
    """
    results = []
    for entry in data_test:
        question = entry["question"]
        reference_answer = entry["answer"]

        # Retrieve context and answer using GraphRAG
        retrieved_contexts = GraphRAG.get_graphRAG_context(question)
        generated_answer = GraphRAG.answer_graphRAG(question)

        results.append({
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "response": generated_answer,
            "reference": reference_answer,
        })

    eval_dataset = EvaluationDataset.from_list(results)
    evaluation_result = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            ContextRecall(),
            ResponseRelevancy(),
            Faithfulness(),
            SemanticSimilarity(embeddings=embedding_wrapper)
        ],
        llm=llm_wrapper
    )

    #evaluation_result.upload()
    return evaluation_result


def evaluate_RAG(data_test: list):
    """
    Evaluate the performance of the standard RAG.

    Args:
        data_test: List of dictionaries with 'question' and 'answer'.

    Returns:
        EvaluationResult object from RAGAS.
    """
    results = []
    for entry in data_test:
        question = entry["question"]
        reference_answer = entry["answer"]

        # Retrieve context and answer using standard HybridRetriever RAG
        retrieved_contexts = GraphRAG.get_RAG_context(question)
        generated_answer = GraphRAG.answer_RAG(question)

        results.append({
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "response": generated_answer,
            "reference": reference_answer,
        })

    eval_dataset = EvaluationDataset.from_list(results)
    evaluation_result = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            ContextRecall(),
            ResponseRelevancy(),
            Faithfulness(),
            SemanticSimilarity(embeddings=embedding_wrapper)
        ],
        llm=llm_wrapper
    )

    #evaluation_result.upload()
    return evaluation_result
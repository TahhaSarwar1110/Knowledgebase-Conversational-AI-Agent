#rag/chain.py
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def create_rag_chain(retriever):
    """
    Create a RAG (Retrieval-Augmented Generation) chain with conversation memory.
    """
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )

        # Create memory to store conversation summaries
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )

        # Create the Conversational Retrieval Chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

        return rag_chain, memory

    except Exception as e:
        print(f"Error in create_rag_chain: {e}")
        raise

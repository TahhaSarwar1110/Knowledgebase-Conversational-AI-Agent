# rag/chain.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

def create_rag_chain(retriever):
    """
    Create a RAG chain with conversation memory
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create memory for conversation history
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create RAG chain
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

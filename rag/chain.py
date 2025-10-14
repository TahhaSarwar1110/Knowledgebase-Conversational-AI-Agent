# rag/chain.py
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.runnables import chain

TEMPLATE = """
The AI should:
1. Respond strictly based on the provided context — only use information available in the given context.
2. Avoid hallucinating or making assumptions beyond the context.
3. Provide clear, complete, and structured answers that help the user take action easily.
4. If the context doesn't provide an answer, reply with: "Sorry, I don't know the answer."
5. If the question lacks enough detail, ask the user politely for more specific information before answering.
6. If the user says phrases like "thank you", "thanks", "ok bye", "bye", "that’s all", or similar — 
  AI should ask a follow up question about any other question and  then gracefully ends the conservation with a short, polite closing such as:
   - "Most welcome! Feel free to reach out anytime 😊"
   - "You're most welcome! Happy to help — feel free to reach out again anytime."
   - "Glad I could assist! Don’t hesitate to reach out whenever you need support."
7. Maintain a professional, respectful, and friendly tone throughout the chat.
8. Avoid unnecessary repetition or filler responses.

Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
"""
def create_rag_chain(retriever):
    """
    Create a RAG chain with conversation memory
    """
    # Chatbot memory
    chat_memory = ConversationSummaryMemory(
        llm=ChatOpenAI(),
        memory_key='message_log'
    )
    prompt_template = PromptTemplate.from_template(template=TEMPLATE)
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        max_tokens=250,
    )
    # Chain combining memory and RAG
    @chain
    def memory_rag_chain(question):
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        # If no relevant context found
        if not context.strip():
            response = "Sorry, I don't know the answer."
            chat_memory.save_context(
                inputs={'input': question},
                outputs={'output': response}
            )
            return response
        # Combining memory and RAG for the prompt
        chain = (
            RunnablePassthrough.assign(
                message_log=RunnableLambda(chat_memory.load_memory_variables) | itemgetter("message_log"),
                context=RunnablePassthrough()
            )
            | prompt_template
            | chat
            | StrOutputParser()
        )
        # Invoke the chain
        response = chain.invoke({'question': question, 'context': context})
        # Save interaction in memory
        chat_memory.save_context(
            inputs={'input': question},
            outputs={'output': response}
        )
        return response
    return memory_rag_chain, chat_memory

   

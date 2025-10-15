from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.runnables import chain

TEMPLATE = """
The AI should:

Answer based on the provided context: Only use the information available in the provided context to respond to questions.
Avoid hallucinating: The AI should not make up or speculate answers beyond the provided context.
Provide a complete, clear, and well-structured response, enhancing overall user experience: The AI should summarize the answer but cover all necessary steps or information for the user to act on the response.
If the context doesn't provide an answer,The AI should say: Sorry, I don't know the answer.
if not enough details are provided in the question, the AI should ask for more specific information

Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
"""

def create_rag_chain(retriever):
    """Create RAG chain with memory"""
    
    # Chatbot memory
    chat_memory = ConversationSummaryMemory(
        llm=ChatOpenAI(), 
        memory_key='message_log'
    )
    
    prompt_template = PromptTemplate.from_template(template=TEMPLATE)
    
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
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

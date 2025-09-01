from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.runnables import chain


from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ ERROR: OPENAI_API_KEY is missing.")


page = PyPDFLoader('User Query_2.pdf')
my_document = page.load()

#header_splitter = MarkdownHeaderTextSplitter(
   # headers_to_split_on=[("#", "Course Title"), ("##", "Lecture Title")]
#)
#header_splitted_document = header_splitter.split_text(my_document[0].page_content)
#for i in range(len(header_splitted_document)):
    #header_splitted_document[i].page_content = ' '.join(header_splitted_document[i].page_content.split())

character_splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=50)
character_splitted_documents = character_splitter.split_documents(my_document)
for i in range(len(character_splitted_documents)):
    character_splitted_documents[i].page_content = ' '.join(character_splitted_documents[i].page_content.split())

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = Chroma.from_documents(
    embedding=embedding,
    documents=character_splitted_documents,
    persist_directory="./TechnoSurge"
)

retriever = vector_store.as_retriever(search_type='similarity_score_threshold', search_kwargs={'k': 1, 'score_threshold': 0.8})
    

#chatbot memory
chat_memory = ConversationSummaryMemory(llm=ChatOpenAI(), memory_key='message_log')

#prompt template for the chatbot
TEMPLATE = """
The AI should:

Answer strictly based on the provided context: Only use the information available in the provided context to respond to questions.
Avoid hallucinating: The AI should not make up or speculate answers beyond the provided context.
Provide a complete, clear, and well-structured response, enhancing overall user experience: The AI should summarize the answer but cover all necessary steps or information for the user to act on the response.
If the context doesn’t provide an answer,The AI should say: Sorry, I don't know the answer.
if not enough details are provided in the question, the AI should ask for more specific information


Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
"""
prompt_template = PromptTemplate.from_template(template=TEMPLATE)

# ChatOpenAI instance
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=250,
)
#chain combining memory and RAG
@chain
def memory_rag_chain(question):
    # Retrieve relevant documents from the vector store
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # If no relevant context is found, a "Sorry" response is returned
    if not context.strip():
        response = "Sorry, I don't know the answer."
        chat_memory.save_context(inputs={'input': question}, outputs={'output': response})
        return response

    # Combining memory and RAG for the prompt
    chain = (
        RunnablePassthrough.assign(
            message_log=RunnableLambda(chat_memory.load_memory_variables) | itemgetter("message_log"),
            context=RunnablePassthrough()  # Pass the RAG context
        )
        | prompt_template
        | chat
        | StrOutputParser()
    )

    # Invoking the chain
    response = chain.invoke({'question': question, 'context': context})

    # Saving the interaction in memory
    chat_memory.save_context(inputs={'input': question}, outputs={'output': response})

    return response

# Usage
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = memory_rag_chain.invoke(user_input)
        print(f"Taha's Bot: {response}")


# %%
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma

# from langchain.chains.question_answering import load_qa_chain ## costly
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings

# %%

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "../db/vectorStore/chromadb"
vectorStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# %%
question = "References and Further Reading"
question = "regression"
matching_docs = vectorStore.similarity_search(question, k=3)
print(matching_docs)

# %%
# ## Make a chain
#
# llm parameter is set to OpenAI(), which is the base class for all OpenAI models. This means that the RetrievalQA model will use a generic OpenAI model, without any specific hyperparameters or focus.
#

# %%
# openai.api_key = os.getenv("OPENAI_API_KEY")

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0)
# print(llm.predict("Hello Bot!"))
# %%
# ## Make a retriever
vectorStoreRetriever = vectorStore.as_retriever(search_kwargs={"k": 3})
# %%


# # Build prompt
# from langchain.prompts import PromptTemplate

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template=template,
# )
# # %%
# # Run chain
# # question = "what is random forest"

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorStoreRetriever,
#     chain_type="stuff",
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
# )

# result = qa_chain({"query": question})
# print(result)
# print(result["result"])


# %% conversational Retrieval Chain
from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorStoreRetriever,
    # memory=memory,
    chain_type="stuff",
    return_source_documents=True,
    return_generated_question=True,
)

chat_history = []
question = "what is the iban of WORLD FUEL SERVICE SINGAPORE5"
result = conversational_chain({"question": question, "chat_history": chat_history})
# result = conversational_chain({"question": question})
print(result)
print(result["answer"])

# %%
chat_history.append((question, result["answer"]))
question = "can you explain more about it?"

result = conversational_chain({"question": question, "chat_history": chat_history})
print(result)
print(result["answer"])


# %%
## Cite sources
def process_llm_response(llm_response):
    print(llm_response["answer"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


# %%
# example
question = "what is the iban of WORLD FUEL SERVICE SINGAPORE5"
llm_response = conversational_chain(
    {"question": question, "chat_history": chat_history}
)
process_llm_response(llm_response)

# %%
# break it down
question = "hi?"
llm_response = conversational_chain(
    {"question": question, "chat_history": chat_history}
)
process_llm_response(llm_response)

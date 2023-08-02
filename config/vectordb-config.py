# %% [markdown]
# ## Setting up LangChain
# conda env: openai
# 

# %%
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS


openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Model.list()

# %% [markdown]
# ## Load multiple and process documents
# 

# %%
# Load and process the text files
loader = DirectoryLoader("../docs", glob="./*.pdf", loader_cls=PyPDFLoader)
pages = loader.load_and_split()

documents = loader.load()
# %%
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splited_docs = text_splitter.split_documents(documents)
    return splited_docs


splited_docs = split_docs(documents)
print(len(splited_docs))
 

# %%
## here we are using OpenAI embeddings
embeddings = OpenAIEmbeddings()
# %% [markdown]
# ## create the DB
# 

# %%
# Embed and store the docs

vectordb = FAISS.from_documents(documents, embeddings)

query = "invoice date"
matching_docs = vectordb.similarity_search(query)
matching_docs

# %%
# The returned distance score is L2 distance. Therefore, a lower score is better.
matching_docs_and_scores = vectordb.similarity_search_with_score(query)

matching_docs_and_scores

# %% [markdown]
# ## Saving
# %%
directory = '../db/vectordb/faiss_index'

vectordb.save_local(directory)

# %% [markdown]
# ## Loading

# %%
new_db = FAISS.load_local(directory, embeddings)
# %% [markdown]
# ## Make a retriever
# 

# %%
retriever = vectordb.as_retriever()

# %%
docs = retriever.get_relevant_documents("who is the invoice receiver?")
docs

# %%
len(docs)

# %%
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# %%
retriever.search_type

# %%
retriever.search_kwargs

# %% [markdown]
# ## Make a chain
# 
# llm parameter is set to OpenAI(), which is the base class for all OpenAI models. This means that the RetrievalQA model will use a generic OpenAI model, without any specific hyperparameters or focus.
# 

# %%
from langchain.chat_models import ChatOpenAI

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0)

qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

query = "qual è amount e quantity di l'airport fee e quale pagina e quale documento è la risposta?"

matching_docs = vectordb.similarity_search(query)
print(matching_docs)
answer = qa_chain.run(input_documents=matching_docs, question=query)
answer

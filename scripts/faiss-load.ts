import { FaissStore } from 'langchain/vectorstores/faiss';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const filePath = 'docs';
const directoryLoader = new DirectoryLoader(filePath, {
  '.pdf': (path) => new PDFLoader(path),
});
// Create docs with a loader
const rawDocs = await directoryLoader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await textSplitter.splitDocuments(rawDocs);

// Load the docs into the vector store
const embedder = new OpenAIEmbeddings();
const vectorStore = await FaissStore.fromDocuments(docs, embedder);

// Search for the most similar document
const resultOne = await vectorStore.similaritySearchWithScore('date', 3);
console.log(resultOne);

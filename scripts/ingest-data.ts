import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { FaissStore } from 'langchain/vectorstores/faiss';

const filePath = 'docs';
const directory = '/db/vectordb';

export const run = async () => {
  try {
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

    // Save the vector store
    await vectorStore.save(directory);

    // // Search for the most similar document
    // const resultOne = await vectorStore.similaritySearchWithScore('date', 3);
    // console.log(resultOne);
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();

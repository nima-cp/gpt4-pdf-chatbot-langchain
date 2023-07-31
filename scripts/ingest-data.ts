import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { Chroma } from 'langchain/vectorstores/chroma';
import { COLLECTION_NAME } from '@/config/chroma';

import {
  JSONLoader,
  JSONLinesLoader,
} from 'langchain/document_loaders/fs/json';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CSVLoader } from 'langchain/document_loaders/fs/csv';
import { DocxLoader } from 'langchain/document_loaders/fs/docx';
import { UnstructuredLoader } from 'langchain/document_loaders/fs/unstructured';

/* Name of directory to retrieve your files from */
const filePath = 'docs';

export const run = async () => {
  try {
    /*load raw docs from the all files in the directory */
    const directoryLoader = new DirectoryLoader(filePath, {
      '.pdf': (path) => new PDFLoader(path),
      '.docx': (path) => new DocxLoader(path),
      '.json': (path) => new JSONLoader(path, '/texts'),
      '.jsonl': (path) => new JSONLinesLoader(path, '/html'),
      '.txt': (path) => new TextLoader(path),
      '.csv': (path) => new CSVLoader(path, 'text'),
      '.htm': (path) => new UnstructuredLoader(path),
      '.html': (path) => new UnstructuredLoader(path),
      '.ppt': (path) => new UnstructuredLoader(path),
      '.pptx': (path) => new UnstructuredLoader(path),
    });

    const rawDocs = await directoryLoader.load();

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    // console.log('split docs', docs);

    console.log('creating vector store...');
    /*create and store the embeddings in the vectorStore*/

    const embedder = new OpenAIEmbeddings();

    await Chroma.fromDocuments(docs, embedder, {
      collectionName: COLLECTION_NAME,
    });
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();

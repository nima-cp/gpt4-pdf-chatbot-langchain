import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { Chroma } from 'langchain/vectorstores/chroma';
import { COLLECTION_NAME } from '@/config/chroma';

// import { PineconeStore } from 'langchain/vectorstores/pinecone';
// import { pinecone } from '@/utils/pinecone-client';
// import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
// import {HttpsProxyAgent} from "https-proxy-agent";

/* Name of directory to retrieve your files from 
   Make sure to add your PDF files inside the 'docs' folder
*/
const filePath = 'docs';

export const run = async () => {
  try {
    /*load raw docs from the all files in the directory */
    const directoryLoader = new DirectoryLoader(filePath, {
      '.pdf': (path) => new PDFLoader(path),
    });

    // const loader = new PDFLoader(filePath);
    const rawDocs = await directoryLoader.load();

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log('split docs', docs);

    console.log('creating vector store...');
    /*create and store the embeddings in the vectorStore*/
    const embeddings = new OpenAIEmbeddings();
    // const agent = new HttpsProxyAgent('http://IP:8080');
    // const embeddings = new OpenAIEmbeddings({ verbose: true, openAIApiKey: '' }, { baseOptions: { proxy: false, httpAgent: agent, httpsAgent: agent, }, });
    // const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name

    let chroma = new Chroma(embeddings, { collectionName: COLLECTION_NAME });
    await chroma.index?.reset();
    //embed the PDF documents
    // await PineconeStore.fromDocuments(docs, embeddings, {
    //   pineconeIndex: index,
    //   namespace: PINECONE_NAME_SPACE,
    //   textKey: 'text',
    // });

    // Ingest documents in batches of 100
    for (let i = 0; i < docs.length; i += 100) {
      const batch = docs.slice(i, i + 100);
      await Chroma.fromDocuments(batch, embeddings, {
        collectionName: COLLECTION_NAME,
      });
    }
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();

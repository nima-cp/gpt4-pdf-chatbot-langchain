import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { makeChain } from '@/utils/makechain';
import { COLLECTION_NAME } from '@/config/chroma';
import { Chroma } from 'langchain/vectorstores/chroma';
import { BaseMessage, HumanMessage, AIMessage } from 'langchain/schema';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  let histories: BaseMessage[] = [];
  history.forEach((hist) => {
    if (hist['type'] === 'human') {
      let req: BaseMessage = new HumanMessage(question);
      histories.push(req);
    } else if (hist['type'] === 'ai') {
      let respond: BaseMessage = new AIMessage(question);
      histories.push(respond);
    }
  });

  console.log('question:', question);

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    /* create vectorStore*/
    const vectorStore = await Chroma.fromExistingCollection(
      new OpenAIEmbeddings({}),
      {
        collectionName: COLLECTION_NAME,
      },
    );

    //create chain
    const chain = makeChain(vectorStore);
    //Ask a question using chat history
    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: histories || [],
    });

    console.log('response', response);
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error:', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}

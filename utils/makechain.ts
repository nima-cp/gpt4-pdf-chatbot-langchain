import { OpenAI } from 'langchain/llms/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { Chroma } from 'langchain/vectorstores/chroma';

interface ResponseType {
  short: number;
  normal: number;
  long: number;
}
const type_of_response: ResponseType = {
  short: 0.0,
  normal: 0.5,
  long: 1,
};
const temperature: number = type_of_response.short;

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful multilingual AI assistant. Use the following pieces of context to answer in the same language as the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorStore: Chroma) => {
  const model = new OpenAI({
    temperature: temperature, // increase temperature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    // modelName: 'gpt-3.5-turbo-0613',
    //modelName: 'gpt-4',
  });
  const vectorStoreRetriever = vectorStore.asRetriever();
  vectorStoreRetriever.k = 2;
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStoreRetriever,
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};

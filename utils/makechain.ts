import { OpenAI } from 'langchain/llms/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { Chroma } from 'langchain/vectorstores/chroma';

const CONDENSE_PROMPT = `Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up in modo che sia una domanda a sé stante.

Cronologia chat:
{chat_history}
Input di follow-up: {question}
Domanda autonoma:`;

const QA_PROMPT = `Sei un utile assistente AI. Usa i seguenti elementi di contesto per rispondere alla domanda alla fine.
Se non conosci la risposta, dì semplicemente che non lo sai. NON cercare di inventare una risposta.
Se la domanda non è correlata al contesto, rispondi educatamente che sei sintonizzato per rispondere solo a domande correlate al contesto.

{context}

Domanda: {question}
Risposta utile in markdown:`;

export const makeChain = (vectorstore: Chroma) => {
  const model = new OpenAI({
    temperature: 0, // increase temperature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    // modelName: 'gpt-3.5-turbo-0613',
    //modelName: 'gpt-4',
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};

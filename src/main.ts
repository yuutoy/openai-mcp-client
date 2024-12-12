import OpenAI from "npm:openai@4.76.1";
import { mcpClient } from "./client.ts";
import {
  applyToolCallIfExists,
  isDone,
  mapToolListToOpenAiTools,
} from "./openai-utils.ts";
import { OPENAI_API_KEY, OPENAI_MODEL } from "./env.ts";
import { MessageHandler, type MessageType } from "./messages.ts";
import { performNextStepSystemPrompt } from "./prompts.ts";
import { askForInput } from "./cli.ts";

const mainLoop = async (messagesHandler: MessageHandler) => {
  const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
  });

  const tools = await mcpClient.listTools();

  // Maximum number of autonomous steps
  const maxIterations = 10;

  for (let i = 0; i < maxIterations; i++) {
    const response = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.2,
      messages: messagesHandler.getMessages(),
      tools: mapToolListToOpenAiTools(tools),
    });

    messagesHandler.addMessage(response.choices[0].message);

    if (isDone(response)) {
      break;
    }

    const toolCallResponse = await applyToolCallIfExists(response);

    if (toolCallResponse.length) {
      messagesHandler.addMessages(toolCallResponse);
    }

    messagesHandler.addMessage(performNextStepSystemPrompt);
  }
};

const messagesHandler = new MessageHandler();

try {
  while (true) {
    const input = await askForInput();

    if (input === "exit") {
      messagesHandler.storeMessages();
      break;
    }

    messagesHandler.addMessage({
      role: "user",
      content: input,
    } as MessageType);

    await mainLoop(messagesHandler);
  }
} catch (error) {
  console.error(error);
  messagesHandler.storeMessages();
} finally {
  mcpClient.close();
}

mcpClient.close();

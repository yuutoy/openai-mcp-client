import OpenAI from "npm:openai@4.76.1";
import { callTool } from "./call-tool.ts";
import { mcpClient } from "./client.ts";
import type { MessageType } from "./messages.ts";

type OpenAiToolsInputType = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>; // JSONSchema
  };
};

type ToolsListServerResponseType = {
  tools: {
    name: string;
    description?: string;
    inputSchema: Record<string, unknown>; // JSONSchema
  }[];
};

/**
 * Maps the tool list received from the server via tools/list to the OpenAI tools format
 * @param toolList
 * @returns
 */
export const mapToolListToOpenAiTools = (
  toolList: ToolsListServerResponseType
): OpenAiToolsInputType[] => {
  return toolList.tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
};

/**
 * Applies the tool call(s) if they exists in the response and returns the result as a message to append
 * @param response
 * @returns
 */
export const applyToolCallIfExists = async (
  response: OpenAI.Chat.Completions.ChatCompletion
): Promise<MessageType[]> => {
  if (!response.choices?.[0]?.message?.tool_calls?.length) {
    return [];
  }

  const toolCallResults: MessageType[] = [];

  for (const toolCall of response.choices[0].message.tool_calls) {
    const toolCallId = toolCall.id;
    const { name, arguments: args } = toolCall.function;

    const [err, result] = await callTool(mcpClient, name, args);

    // console.log("Got result from tool call:", result);

    if (err) {
      // console.log("An error occurred while calling the tool:", err);
      throw err;
    }

    if (!result.content?.length) {
      // console.log("No content returned from tool");
      throw new Error("No content returned from tool");
    }

    switch (result.content[0].type) {
      case "text":
        toolCallResults.push({
          role: "tool",
          content: result.content[0].text,
          tool_call_id: toolCallId,
        });
        break;
      default:
        // console.log("Unknown content type returned from tool:", result.content);
        throw new Error(
          "Unknown content type returned from tool:" + result.content
        );
    }
  }

  return toolCallResults;
};

export const isDone = (
  response: OpenAI.Chat.Completions.ChatCompletion
): boolean => {
  if (!response.choices?.length) {
    // console.log("No choices found in response");
    throw new Error("No choices found in response");
  }

  const choice = response.choices[0];

  return choice.finish_reason === "stop";
};

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
import torch

APIKey = "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"

if __name__ == '__main__':
    llm = OpenRouter(
        api_key=APIKey,
        max_tokens=256,
        context_window=1024,
        #model="google/gemma-3n-e2b-it:free"
        model="mistralai/mistral-7b-instruct-v0.1"
    )
    query_l = ["xCan a creature cast spells while grappled?",
               "!Can I as a Cleric still cast Spells if i carry a shield instead of my Holy Symbol?",
               "oCan a creature make opportunity attacks while prone?",
               "xCan a character cast two leveled spells in the same turn?",
               "x!Can a character cast a reaction spell in the same round as a bonus-action spell?",
               "xCan Counterspell be used to stop another Counterspell?",
               "xDoes the prone condition give me disadvantage on ranged spell attacks?",
               "x!Can a creature move between attacks during the Extra Attack feature?",
               "x!If i am shoved, do I trigger the Effect of Booming Blade?",
               "oCan you take reactions while stunned?"
               ]
    for q in query_l:
        query = f"In DnD 5e: {q}"
        response_llm = llm.complete(query)
        print(query)
        print("==============================================================")
        print(f"Complete Response:\n{response_llm}")
    #message = ChatMessage(role="user", content=query)
    #response_chat = llm.chat([message])
    #print(f"Chat Response:\n{response_chat}")
    #query_engine = llm.as_query_engine()
    #response_engine = query_engine.run(query)
    #print(response_engine)

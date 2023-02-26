from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_chain():
    """Logic for loading the chain you want to use should go here."""

    model_id = "facebook/blenderbot_small-90M"
    model_id = "google/flan-T5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline=pipe)

    input_key = "input"
    output_key = "response"
    memory = ConversationalBufferWindowMemory(k=3, output_key=output_key, input_key=input_key)
    chain = ConversationChain(
        llm=llm, verbose=True, memory=memory, output_key=output_key, input_key=input_key
    )
    return chain
from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

def load_chain():
    """Logic for loading the chain you want to use should go here."""

    model_id = "facebook/blenderbot_small-90M"
    model_id = "google/flan-T5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline=pipe)

    chain = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationSummaryBufferMemory(
            llm=llm, output_key="response", input_key="input"
        ),
    )
    return chain

# Validated Answer Generator with LangChain

This Python module is to demonstrate a solution for generating validated answers to customer questions with large
language models (LLMs) using LangChain, a versatile language model framework. This module leverages LangChain's
sequential LLMChains to create a structured process for generating answers and validating them against predefined
criteria.

## Features

- **SequentialChain Structure**: Utilizes LangChain's SequentialChain to organize the answer generation and validation
  process.
- **LLMChain Integration**: Incorporates LangChain's LLMChain to interface with large language models (LLMs) for natural
  language understanding.
- **Input Validation**: Validates generated answers based on specified criteria using LangChain's prompts and chains.
- **Extensibility**: Easily extendable for different use cases by inheriting from the base `ValidatedAnswerGenerator`
  class.

## Usage

1. **Initialization**: Create an instance of `ValidatedAnswerGenerator` or its subclass, `OrderAnswerGenerator`,
   providing the necessary Language Model (LLM), prompts, and validation criteria.

   ```python
   from langchain.chains import SequentialChain, LLMChain
   from langchain.llms.base import LLM
   from langchain.prompts import PromptTemplate

   # Example initialization
   llm = YourLanguageModel()
   prompt = YourPromptTemplate()
   validation_prompt = YourValidationPromptTemplate()

   generator = ValidatedAnswerGenerator(llm, prompt, validation_prompt)

This project uses llama-cpp as an interface to utilize the LLama2 Chat LLM.

## Example test results

Let's look at the test prompt with the question input:

```
When my order will arrive? I forgot, how many items it has?
```

LangChain will run this prompt with the input context provided, then the generated answer will be the input of the
second prompt, which is for validating the answer against a set of criteria, in the below example by specifying that
no "negativity, sarcasm, offensive remark" should appear in the answer from the first prompt. Only an answer which
fulfils the validation criteria would be considered as valid that can be surfaced to the customer.

Let's see an example that was generated for a delivery order containing 7 items and with 23-minute ETA:

```
Hooray! Your order is on its way! 🎉 You've got 7 items in your grocery haul, and it's estimated to arrive in 23 minutes! 🕰️ You can track the status of your order on MyMart Online or in the MyMart app. ✨ Don't forget to check out the tacking screen for updates! 👉✌️
```

The full evaluation with the order and validation criteria inputs:

```
{
  "order": {
    "ID": "DGF-349",
    "delivery courier": "Uber",
    "weight": "3100 gr",
    "items": "7",
    "ETA": "23 m",
    "KYC required": "false"
  },
  "provider": "Uber",
  "heavy_limit": 5000,
  "assistant_style": "cheerful",
  "validation_criteria": "negativity, sarcasm, offensive remark",
  "question": "When my order will arrive? I forgot, how many items it has?",
  "answer": "  Hooray! Your order is on its way! \ud83c\udf89 You've got 7 items in your grocery haul, and it's estimated to arrive in 23 minutes! \ud83d\udd70\ufe0f You can track the status of your order on MyMart Online or in the MyMart app. \u2728 Don't forget to check out the tacking screen for updates! \ud83d\udc49\u270c\ufe0f",
  "validation_result": "  No, this text does not have any negativity, sarcasm, or offensive remarks. It is a positive and informative message indicating that an order is on its way and providing details about the estimated arrival time and how to track the status of the order. The use of emojis and exclamation points adds a friendly and celebratory tone to the message.",
  "is_valid": true
}
```

In a second case, when the order has 9 items and a delivery ETA of 28 minutes, we expect that the answer will reflect
that:

```
Hooray! Your order is on its way! 🛍️🚲 According to the details you provided, your order has 9 items and weighs 5200 gr. 💪 It's being delivered via Quickfox scooter, and you can track its progress on MyMart Online or in the MyMart app. The estimated arrival time is 28 minutes from now! ⏰

So, sit back, relax, and get ready to receive your groceries soon! If you have any more questions or need further assistance, just give me a cheerful holler! 😊
```

The full evaluation with order and validation criteria inputs for this second case:

```
{
  "order": {
    "ID": "ATH-277",
    "delivery courier": "Quickfox",
    "weight": "5200 gr",
    "items": "9",
    "ETA": "28 m",
    "KYC required": "true"
  },
  "provider": "Quickfox",
  "heavy_limit": 5000,
  "assistant_style": "cheerful",
  "validation_criteria": "negativity, sarcasm, offensive remark",
  "question": "When my order will arrive? I forgot, how many items it has?",
  "answer": "  Hooray! Your order is on its way! \ud83d\udecd\ufe0f\ud83d\udeb2 According to the details you provided, your order has 9 items and weighs 5200 gr. \ud83d\udcaa It's being delivered via Quickfox scooter, and you can track its progress on MyMart Online or in the MyMart app. The estimated arrival time is 28 minutes from now! \u23f0\n\nSo, sit back, relax, and get ready to receive your groceries soon! If you have any more questions or need further assistance, just give me a cheerful holler! \ud83d\ude0a",
  "validation_result": "  No, this text does not contain any negativity, sarcasm, or offensive remarks. It is a positive and helpful message, with a friendly and enthusiastic tone.",
  "is_valid": true
}
```

## Local setup

### Setting up the LLM CLI & llama-cpp

A useful way to set up and interact with LLMs locally is the [LLM CLI](https://llm.datasette.io/en/stable/help.html)
tool, which can be installed like:

```angular2html
pip install llm
```

Or on Mac you might prefer to do this step with homebrew:

```
brew install llm
```

Then you will need the llm-llama-cpp plugin which adds support for Llama-style models, building on top of the
llama-cpp-python bindings for llama.cpp.
Installing this plugin takes two steps. The first is to install the plugin itself:

```
llm install llm-llama-cpp
```

Need to install the llama-cpp-python bindings as well. One way to do this is by running:

```
llm install llama-cpp-python
```

### Obtaining a model to run locally

You can download a suitable Llama2 Chat model from [Huggingface](https://huggingface.co/).

One model that can be used with llama-cpp directly
is [Llama-2-13B-chat-GGUF](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)

If you want to use it with llama-cpp CLI, you will have to add the model using the _add-model_ command like so:

```angular2html
llm llama-cpp add-model /Users/aa/llm_models/llama-2-13b-chat.Q6_K.gguf --alias llama2-chat-13B --alias l2c13b --llama2-chat
```

### Specifying LLM model path

LangChain requires to specify the path the local model can be accessed.

For running the tests, this can be done by setting an LLM_MODEL_PATH environment variable with the full path of the
model as the
value.

For example with the model from Huggingface described above:

```
LLM_MODEL_PATH=/Users/aa/llm_models/llama-2-13b-chat.Q6_K.gguf
```



# Natural Language Query Agent

The primary objective of this project is to develop a Natural Language Query Agent that leverages Large Language Models (LLMs) and open-source vector indexing and storage frameworks to provide concise responses to straightforward queries within a substantial dataset comprising lecture notes and a table detailing LLM architectures. While the fundamental requirement is to generate articulate responses based on reference text, this project will encompass strategies and concepts for extending the agent's capabilities to address more intricate inquiries, including follow-up questions, conversational memory, reference citation, and the ability to process extensive sets of lecture notes spanning multiple subjects.

The data sources utilized for this project encompass the following:

- [Stanford LLMs Lecture Notes](https://stanford-cs324.github.io/winter2022/lectures/)
- [Awesome LLM Milestone Papers](https://github.com/Hannibal046/Awesome-LLM#milestone-papers)

**Important Note**

> You are required to initiate an access request for Llama 2 models via the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and consent to sharing your [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) account details with Meta. Please ensure that your Hugging Face account email aligns with the one you've submitted on the Meta website, as mismatched emails may lead to request denial. The approval process typically takes a few minutes to a few hours to complete.

**Requirements** 

The following tools are necessary for its proper functionality.

- [transformers](https://pypi.org/project/transformers/)
- [accelerate](https://pypi.org/project/accelerate/)
- [einops](https://pypi.org/project/einops/)
- [langchain](https://pypi.org/project/langchain/)
- [xformers](https://pypi.org/project/xformers/)
- [bitsandbytes](https://pypi.org/project/bitsandbytes/)
- [faiss-gpu](https://pypi.org/project/faiss-gpu/)
- [sentence_transformers](https://pypi.org/project/sentence-transformers/)
- [pypdf](https://pypi.org/project/pypdf/)

**[LLaMA 2](https://ai.meta.com/llama/)** 

The LLaMA 2 model is a powerful open-source language model, pretrained and fine-tuned on an extensive dataset of 2 trillion tokens. It is available in three different sizes: 7B, 13B, and 70B, each offering improvements over the previous LLaMA 1 models. Notable enhancements include training on 40% more tokens, an impressive context length of 4,000 tokens. LLaMA 2 outperforms other open-source language models in various external benchmarks, including tasks related to reasoning, coding, proficiency, and knowledge assessment.

**[LangChain](https://python.langchain.com/docs/get_started/introduction)** 

LangChain is a robust open-source framework tailored for building applications driven by language models, especially large ones. The key concept behind this library is the ability to interconnect various components to enable more advanced functionalities centered around large language models (LLMs). LangChain comprises multiple components spread across different modules.

**[FAISS](https://github.com/facebookresearch/faiss)**


FAISS (Facebook AI Similarity Search) is a library that empowers developers to swiftly find similar embeddings in multimedia documents. It overcomes constraints of conventional hash-based query search engines, offering scalable similarity search capabilities. Utilizing FAISS, developers can efficiently search multimedia documents, even in ways that are impractical with standard SQL database engines. It incorporates nearest-neighbor search solutions tailored for datasets ranging from millions to billions in scale, effectively balancing memory, speed, and accuracy considerations. FAISS is designed to excel in performance across a wide range of operational scenarios.

This project is divided into three essential parts.
- **Preparing the dataset**
- **Initializing the HuggingFace text-generation pipeline**
- **Initializing the conversational chain**

### Preparing the Dataset

A big challenge in this project is preparing the dataset. Preparing the dataset by retrieving the text manually is time-consuming and not scalable. Writing an automated script to read HTML documents may not be exhaustive enough to cover all of the cases. Fortunately, LangChain has document readers which can directly retrieve and process text from web links and pdf links. We will utilize the document loaders provided by Langchain, which include:

- [WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base): WebBaseLoader represents a robust framework designed for extracting text content from HTML web pages and transforming it into a document format suitable for a wide range of downstream tasks.
- [PyPDF](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf): PyPDF is a potent framework crafted for the extraction of text from PDF files.

We will retrieve links to webpages and PDF files from the following .txt documents:

	ema_dataset_web_links.txt
	ema_dataset_pdf_links.txt

At the end of the process, we'll preserve the custom dataset as a .pkl file, an integral part of our pipeline. You may refer to the `ema_dataset.ipynb` notebook present in the `ema-dataset` folder for the exact implementation

### Initializing the HuggingFace text-generation Pipeline

To set up a text-generation pipeline using Hugging Face transformers, you must initiate the following three vital elements:

- **A Language Model (LLM)**: We will utilize `meta-llama/Llama-2-7b-chat-hf`
- **A Tokenizer**: The pipeline necessitates a tokenizer responsible for translating human-readable text into token IDs readable by the LLM. The Llama 2 7B models were trained using the Llama 2 7B tokenizer
- **A stopping criteria object**: It's important to establish the stopping criteria for the model. Stopping criteria help determine when the model should cease generating text. Without clear criteria in place, the model might continue producing text that deviates from the initial question.

  
### Initializing the Conversational Chain

The next step is the initialization of the `ConversationalRetrievalChain`. This chain empowers you to create a chatbot with a memory feature, leveraging a vector store to retrieve pertinent information from your documents

Moreover, if you wish to retrieve the source documents utilized to address a question, you can do so by specifying an optional parameter, specifically setting `return_source_documents` to `True` during the chain's construction.

	chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
	
You may refer to to the `ema-QA-model.ipynb` notebook present in the `ema-QA-model` folder.

### Final Results

The ultimate model boasts impressive capabilities, including the generation of articulate responses from reference texts. It excels in managing complex queries, encompassing follow-up questions, conversational memory, reference citations, and the processing of extensive sets of lecture notes across various subjects.

- #### **Question-Answering and Reference Citation**

The conversation below was generated with the final model. Please make note of the fact that the model is capable of citing references to demonstrate that the model is not hallucinating.
	
> Enter your Query: What is a language model?
> 
> 
> A language model is a probability distribution over sequences of tokens. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> 
> Enter your Query: What is adaptability?
> 
> 
> The term "adaptable" refers to the ability of a language model to 
> adjust its performance to better suit a particular context or task. This 
> can involve modifying the model's parameters or fine-tuning the model on a 
> small set of in-context examples. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/adaptation/
> 
> 
> Enter your Query: What are some milestone model architectures and papers in the last few years?
> 
> 
> Yes, here are some recent milestone models and papers in natural language 
> processing: * GPT-3 (2020): A large-scale transformer-based language model developed by OpenAI 
> that achieved state-of-the-art results on a wide range of natural language processing 
> tasks. * BERT (2018): A pre-trained language model developed by Google that 
> achieved state-of-the-art results on a wide range of natural language processing tasks, 
> including question answering, sentiment analysis, and text classification. * RoBERTa (2019): A 
> variant of BERT that was specifically designed for text classification tasks and 
> achieved state-of-the-art results on a number of benchmark datasets. * Megatron-LM (2019): 
> A large-scale transformer-based language model developed by Facebook that achieved state-of-the-art results 
> on a wide range of natural language processing tasks, including text classification, 
> sentiment analysis, and question answering. * T5 (2020): A text-to-text transformer model 
> developed by Google that achieved state-of-the-art results on a wide range of 
> natural language processing tasks, including text generation, question answering, and dialogue systems. 
> 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> 
> Enter your Query: What are the layers in a transformer block?
> 
> 
> In the Transformer architecture described in the paper, there are 6 identical 
> layers in the encoder and decoder. Each layer consists of two sub-layers: 
> a multi-head self-attention mechanism and a simple, position-wise fully connected feed-forward network. 
> The output of each sub-layer is then passed through layer normalization and 
> a residual connection before being fed into the next layer. 
> 
> 
> 
> Enter your Query: Tell me about datasets used to train LLMs and how they’re cleaned
> 
> 
> The most common datasets used to train language models are those collected 
> from the internet, such as Common Crawl, and OpenWebText. These datasets are 
> curated and filtered to remove unwanted content, such as adult or offensive 
> material. Additionally, some datasets may undergo tokenization, part-of-speech tagging, and other forms 
> of preprocessing before being used for training. However, I cannot provide more 
> detailed information on the specific datasets used or their cleaning processes without 
> further context. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: thanks

- #### **Conversational Memory**

Furthermore, the model can generate responses by recalling its prior interactions, as illustrated in the following example:

> Enter your Query: What are large language models?
> 
> 
> Large language models are neural network models that are trained on large 
> amounts of text data to generate language outputs that are coherent and 
> natural-sounding. These models have become increasingly popular in recent years due to 
> their ability to perform a wide range of tasks, such as language 
> translation, text summarization, and text generation. They are often used in applications 
> such as chatbots, language translation software, and content creation tools. However, these 
> models also pose significant risks, such as the potential for misuse or 
> unintended consequences, and there are ongoing efforts to develop more responsible and 
> ethical ways of using them. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: Can you please elaborate?
> 
> 
> Of course! Large language models are trained on vast amounts of text 
> data, which allows them to learn patterns and relationships within language. They 
> use various techniques such as word embeddings, attention mechanisms, and neural networks 
> to capture the meaning and context of words and phrases. Through this 
> process, they can generate text, complete sentences, and even engage in conversation. 
> However, it's important to note that these models are not perfect and 
> can sometimes produce inappropriate or offensive content, so it's crucial to use 
> them responsibly and ethically. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: thanks


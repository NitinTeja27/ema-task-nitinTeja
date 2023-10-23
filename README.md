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

The next step is the initialization of the `ConversationalRetrievalChain`. This chain empowers you to create a chatbot with a memory feature, leveraging a vector store to retrieve pertinent information from your documents.

Moreover, if you wish to retrieve the source documents utilized to address a question, you can do so by specifying an optional parameter, specifically setting `return_source_documents` to `True` during the chain's construction.

	chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
	
You may refer to to the `ema-QA-model.ipynb` notebook present in the `ema-QA-model`.

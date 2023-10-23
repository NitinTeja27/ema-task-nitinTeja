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


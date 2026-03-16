## Large-scale collaborative work using statistical and AI tools to collect and organize stem cell differentiation protocols


## Key Capabilities

- Semantic RAG (Retrieval-Augmented Generation)
- Intelligent Summarization using K-Means Clustering
- Structured Protocol Extraction


## Context & Objective

This project was developed as part of my Bachelor's thesis in Biology. The objective 
was to create a tool based on AI and Statistical Tools for the extraction, categorization
and synthesis of stem cell differentiation protocols from large volumes of unstructured 
scientific papers. 


## Usage and Limitations

Store your scientific papers in PDF format in a folder of your choice. Also, make sure 
to add your prefered directories in the "config.py" file. Using that file, you can change 
the question you want to ask based on the paper (PDF) of your choice. Run "main.py" for the
first time to store your papers in the Chromadb. Run it again for CLI applications such as
asking a question or getting a summary.

As of now, the protocol extraction is executed by "protocol.py" from the new_protocol_tools
folder. Indicative results of this can be viewed in the "medic_data" in "protocol_gemma3:12b.txt" 
and "protocol_llama3.1:latest.txt". As you may notice from the .txt names, the difference between 
these two is the llm used for the extraction. Cell line extraction is an exception, where
"gemma3:12b" is hard coded to be used because it consistently detecteted the correct cell line.

The main limitation of this project revolves around two major issues. One is the coding
approach for extracting data related to the protocol. The other is the llm and prompt used
for each step of the extraction process. Both of these were the main challenge of my thesis and
this repository contains my best efforts to tackle them. It is not a fully functional 
end-user application, but rather a proof-of-concept (PoC) that demonstrates the feasibility of 
automated protocol extraction. While the current pipeline achieves significant results in 
identifying key biological parameters, there is still room for improvement regarding 
the accuracy of data parsing and the refinement of the LLM responses to minimize hallucinations. 
This project serves as a foundational framework upon which more robust, production-ready systems 
can be built. 


## Thesis Abstract

This study focuses on the collection and organization of stem cell differentiation protocols
from the international scientific literature, combining human collaboration with artificial
intelligence. An organized team of students extracted critical information from published
protocols, which were stored in a Chromadb database using tools such as Langchain.
Statistical analysis of the data followed, along with the application of AI models for data
extraction and the overall automation of the procedure. This approach aims to facilitate
the management of the ever-growing volume of scientific data and contribute to its more
effective use in biological research.


## 💭 Feedback and Contributing

Feel free to use the discussion tab to open issues for bugs/feature requests or any related
questions:
* https://github.com/e-likoudi/stem_chat/discussions


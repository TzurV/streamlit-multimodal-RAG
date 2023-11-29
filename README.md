# Streamlit Multimodal RAG 

This Streamlit application implements a multimodal Question Answering (QA) system using the LangChain library.

## Key Features

- Interactive Streamlit UI for file uploads, DB build, and QA
- Accepts input files in PDF, audio (WAV, MP3, opus), and text formats
- Transcribes audio to text using HuggingFace DistilWhisper models
- Audio transcription runs in close to real-time on CPU 
- Background loading of models takes time, notice top-right running indicator  
- Requires [HuggingFace API key](https://huggingface.co/docs/hub/security-tokens)
- Docker container exposes port 8001, access UI with browser `localhost:8001`

## Flowchart

<img src="Multimodal RAG.png" width="600">

## Models Used

- Sentence Embeddings: [huggingface/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))
  
  *Note*: input text longer than 384 word pieces is truncated.
- STT: [distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en)
- LLM: [declare-lab/flan-alpaca-large](https://huggingface.co/declare-lab/flan-alpaca-large)


## Installation

```
docker build -t streamlit-app .
docker run -p 8001:8001 --rm streamlit-app
```

GUI access [localhost:8001](http://localhost:8001/)

## Note
Please be aware that this is only a Proof of Concept system and 
may contain bugs or unfinished features.

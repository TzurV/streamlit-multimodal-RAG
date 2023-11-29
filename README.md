# Streamlit Multimodal RAG 

This Streamlit application implements a multimodal Question Answering (QA) system using the LangChain library.

## Key Features

- Accepts input files in PDF, audio (WAV, MP3, opus), and text formats
- Transcribes audio to text using HuggingFace DistilWhisper models
- Audio transcription runs in close to real-time on CPU 
- Background loading of models takes time, notice top-right running indicator  
- Requires HuggingFace API key for access to models
- Interactive Streamlit UI for file uploads, DB build, and QA
- Docker container exposes port 8001, access UI with browser `localhost:8001`

## Flowchart

![Streamlit Multimodal RAG flowchart](<Multimodal RAG.png>)

## Models Used

- STT: [distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en)
- LLM: [declare-lab/flan-alpaca-large](https://huggingface.co/declare-lab/flan-alpaca-large)


## Installation

```
docker build -t streamlit-app .
docker run -p 8001:8001 --rm streamlit-app
```

## Note
Please be aware that this is only a Proof of Concept system and 
may contain bugs or unfinished features.

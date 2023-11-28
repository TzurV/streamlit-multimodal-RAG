
from langchain.vectorstores import Chroma
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

import os, sys
import json
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import my_keys

os.environ["HUGGINGFACEHUB_API_TOKEN"] = my_keys.HUGGINGFACEHUB_API_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def download_transcription_or_audio(youtube_url):
    try:
        # Create a YouTube object
        yt = YouTube(youtube_url)

        # Check if automatic captions (transcription) are available
        if yt.captions:
            # Download the transcription
            caption = yt.captions.get_by_language_code('en')
            transcription_text = caption.generate_srt_captions()
            print("Transcription:\n", transcription_text)

        else:
            # If no captions, download the audio
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_stream.download(output_path='.', filename='video_audio')
            print("Audio downloaded successfully.")

    except Exception as e:
        print(f"Error: {e}")


if True:
    # Example usage
    youtube_url = "https://www.youtube.com/watch?v=UNP03fDSj1U" 
    #download_transcription_or_audio(youtube_url)
    transcript_list = YouTubeTranscriptApi.list_transcripts('UNP03fDSj1U')
    #print(transcript_list)
    transcript = transcript_list.find_transcript(['en'])
    print(transcript.fetch())
    all_text = ''
    for seg in transcript.fetch():
        all_text += seg['text'] + ' '
    print(all_text) 

elif False:
    directory = '/app/txt'
    curent_llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
    loaders = [TextLoader(os.path.join(directory, fn)) for fn in os.listdir(directory)]


    vectorstoreIndex = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

    print(dir(vectorstoreIndex.vectorstore))

    prompt_template = """If the context is not relevant, 
            please answer <I don't know>
            
            {context}
            
            Question: {question}
            """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=curent_llm,
                                chain_type="stuff",
                                retriever=vectorstoreIndex.vectorstore.as_retriever(
                                    search_kwargs={"k": 6}),
                                input_key="question",
                                chain_type_kwargs=chain_type_kwargs)

    question = "What were the main findings from evaluating the proposed DLN on the noisy MSP-Podcast corpus? "
    chain.run(question)


elif:
    # Create an empty list to store the loaded documents
    docs = []


    # Loop through all files in the text directory
    directory = '/app/txt'
    for text_file in os.listdir(directory):
        if text_file.endswith(".txt"):  # Assuming text files have a .txt extension
            # Create the full path to the text file
            text_file_path = os.path.join(directory, text_file)

            # Create a TextLoader for the current text file
            loader = TextLoader(text_file_path)

            # Load the text from the file and append it to the docs list
            loaded_documents = loader.load()
            if loaded_documents:
                docs.extend(loaded_documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    def make_embedder():
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )


    hf = make_embedder()
    db = Chroma.from_documents(texts, hf)

    #print(dir(db))
    #print(f"DB size: {sys.getsizeof(db)}")
    #print(db.get())

    all_documents = db.get()['documents']
    total_records = len(all_documents)
    print("Total records in the collection: ", total_records)

    #print(f"Records indexed: {len(db)}")
    #print(f"Vector dim: {db.vector_size}")

    #metadata = db.get_metadata()
    #print(metadata)

    current_llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})

    prompt_template = """Below is some context. Following the context is a question about it. 
            
            {context}
            
            Question: {question}
                        If the question can be answered from the context, answer it. 
                        If the question cannot be answered from the context, respond with 'I don't know'.

                    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=current_llm,
                                chain_type="stuff",
                                retriever=db.as_retriever(search_kwargs={"k": 6}),
                                input_key="question",
                                chain_type_kwargs=chain_type_kwargs)

    question = "What were the main findings from evaluating the proposed DLN on the noisy MSP-Podcast corpus? "
    print(chain.run(question))

    question = "how many different languages text-to-speech (TTS) solutions focus on synthesizing ?"
    print(chain.run(question))

    question = "What time is it now? "
    print(chain.run(question))



    print("--------- Add PDFs --------------")
    directory = '/app/pdf'
    for file in os.listdir(directory):
        print(file)
    loaders = [UnstructuredPDFLoader(os.path.join(directory, fn))
            for fn in os.listdir(directory)]

    for loader in loaders:
        print(loader)
        loaded_pdf = loader.load()
        #print(loaded_pdf)
        texts = text_splitter.split_documents(loaded_pdf)
        #print('---------------------------- Texts ----------------------------')
        #print(type(texts))
        #print(texts)
        
        print('--- db.add_documents(texts) ---')
        db.add_documents(texts)

        all_documents = db.get()['documents']
        total_records = len(all_documents)
        print("Total records in the collection: ", total_records)





    question = "What time is it now? "
    print(chain.run(question))

    question = "What were the main findings from evaluating the proposed DLN on the noisy MSP-Podcast corpus? "
    print(chain.run(question))
    print(db.similarity_search(question))


    question = "who includes learnable language embeddings?"
    print(chain.run(question))
    print(db.similarity_search(question))

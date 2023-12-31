import streamlit as st
import os
import shutil
from pathlib import Path
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# source: https://github.com/huggingface/distil-whisper
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from youtube_transcript_api import YouTubeTranscriptApi

import torch
import librosa
import numpy as np
import altair as alt
import pandas as pd
from datetime import datetime
import time
import re
import requests
from urllib.parse import urlparse

if os.path.exists("my_keys.py"):
    # Import the module if the file exists
	import my_keys

app_name = "Streamlit Multimodal RAG"

# session state
st.set_page_config(layout='centered', page_title=f'{app_name}')
ss = st.session_state

ss['api_key'] = my_keys.HUGGINGFACEHUB_API_TOKEN

if 'debug' not in ss: ss['debug'] = {}
if 'loaded' not in ss: ss['loaded'] = False
if 'run_return' not in ss: ss['run_return'] = False
if 'transcriber' not in ss: ss['transcriber'] = None
if 'pipe' not in ss: ss['pipe'] = None
if 'qas' not in ss: ss['qas'] = []


def showtime(label=""):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")
    st.write(label, ": Current Time =", current_time)

if ss.run_return:
	time.sleep(1)
ss.run_return = False

class DirectoryStructure:

    def __init__(self):
       self.create_structure()

    def create_structure(self):
        # List of directories to create
        dirs = [
            "pdf",
            "txt",
            "audio",
            #os.path.join("audio", "txt")			
        ]

        # Loop through list and create directories
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

    @property
    def pdf(self):
        return os.path.realpath("pdf")
    
    @property
    def txt(self):
        return os.path.realpath("txt")

    @property
    def audio(self):
        return os.path.realpath("audio")

    #@property
    #def audio_txt(self):
    #    return os.path.realpath(os.path.join("audio", "txt"))

showtime("start")
structure = DirectoryStructure()

curent_llm = None
if 'chain' not in ss: ss['chain'] = None
ss['show_debug'] = False

def set_gf_api_key():

	os.environ["HUGGINGFACEHUB_API_TOKEN"] = ss.get('api_key')

	# load llm
	global curent_llm
	curent_llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":1024})

def load_audio_set_sample_rate(file_path):
	# Load the WAV file using librosa 
	waveform, sample_rate = librosa.load(file_path, sr=None, mono=True) 
    
	if sample_rate != 16000:
		# Resample to 16kHz
		waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    
	# Convert to numpy array
	waveform = np.array(waveform) 

	return waveform, 16000

def ui_debug():
	if ss.get('show_debug'):
		st.write('### debug')
		st.write(ss.get('debug',{}))

def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')

def ui_info():
	st.markdown(f"""
	# {app_name}

    Multimodal Question Answering system supporting 
	urls, pdf, and audio files.

	Thank you for your interest in my application.
	Please be aware that this is only a Proof of Concept system and 
	may contain bugs or unfinished features. 
	""")
	ui_spacer(1)
	st.markdown(
	    'Source code can be found [here](https://github.com/TzurV/streamlit-RAG).')


def ui_question():
	st.write('## 3. Ask questions')
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)

def file_save(uploaded_file):
    
	# Get file name and type
	file_name = Path(uploaded_file.name).stem
	file_ext = Path(uploaded_file.name).suffix.lower()

	# Specify directory
	if file_ext == '.pdf':
		path = structure.pdf
	elif file_ext == '.wav' or file_ext == '.mp3':
		path = structure.audio
	elif file_ext == '.txt':
		path = structure.txt
	else:
		st.write('unknown file type')
		return  

	# Save file to local file system
	local_file = os.path.join(path, f'{file_name}{file_ext}')
	with open(local_file, 'wb') as out_file:
		shutil.copyfileobj(uploaded_file, out_file)

	#st.success(f'Saved PDF to {local_file}')

	# remember
	# Save uploaded file  
	#with open(f"{uploaded_file.name}", "wb") as f: 
	#	f.write(uploaded_file.getbuffer())


def extract_text_from_pdf():
	documents = []
	directory = Path(structure.pdf)
	#for path in directory.glob('*.pdf'):
	#	st.write(f"extracting {path}")

	for fn in os.listdir(directory):
		st.write(f"adding {fn}")
		documents.extend(UnstructuredPDFLoader(os.path.join(directory, fn)).load())

	return documents

def extract_text_from_txt():
	
	directory = Path(structure.txt)
	#for text_file in directory.glob('*.txt'):
	#	st.write(f"adding {text_file}")

	documents = []
	for fn in os.listdir(directory):
		st.write(f"adding {fn}")
		documents.extend(TextLoader(os.path.join(directory, fn)).load())

	return documents

def text_filename_from_original(original_file_name):
	"""
	Generates a text filename from the original file name.

	:param original_file_name: The original file name.
	:type original_file_name: str
	:return: The generated text filename.
	:rtype: str
	"""
	text_filename = Path(structure.txt) / (original_file_name.stem +
                                        '_' + original_file_name.suffix.replace('.', '') + '.txt')
	return Path(text_filename)


def	transcibe_audio():
	directory = Path(structure.audio)
	wav_files = list(directory.glob('*.wav')) 
	mp3_files = list(directory.glob('*.mp3'))
	opus_files = list(directory.glob('*.opus'))

	# source https://github.com/huggingface/distil-whisper
	all_audio_files = wav_files + mp3_files + opus_files
	for audio_file in all_audio_files:

		text_filename = text_filename_from_original(audio_file)
		if text_filename.exists():
			st.write(f"skipping {audio_file} because {text_filename} exists")
			continue

		samples, sample_rate = load_audio_set_sample_rate(audio_file)
		start_time = time.time()
		showtime("Start clock")
		with st.spinner(f"transcribing {audio_file} audio duration {samples.shape[0]/sample_rate:.2f} seconds"):
			#transcriber = ss.transcriber
			#transcription = transcriber(samples)

			transcription = ss.pipe(samples)
			#st.write(transcription)

		elapsed_time = time.time() - start_time 
		st.success('Done!')
		st.write(f"Took {elapsed_time:.2f} seconds")
		
		# save text in file
		with open(text_filename, "w") as f: 
			f.write(transcription['text'])


def download_file(url, local_directory):
	try:
		# Extract the filename from the URL
		filename = os.path.basename(url)
		local_path = os.path.join(local_directory, filename)
		if os.path.exists(local_path):
			st.write(f"File {local_path} already exists")
			return

		response = requests.get(url)
		if response.status_code == 200:

			with open(local_path, 'wb') as file:
				file.write(response.content)
			st.write(f"File downloaded and saved as {local_path}")
			
		else:
			st.write(f"Failed to download. Status code: {response.status_code}")

	except Exception as e:
		st.write(f"An error occurred: {str(e)}")


def process_url(url):
	st.write(f"processing {url}")

	# Check if valid URL 
	if not re.match(r"https?://", url):
		print("Not a valid URL")  
		return
        
	parsed_url = urlparse(url)
	ext = os.path.splitext(parsed_url.path)[1]
    
	if parsed_url.hostname == 'www.youtube.com':
		st.write(f"{url} is a youtube url")
		try:
			video_id = re.search(r"(?:\?v=|\&v=)(.+?)(&|$)", url).group(1)

			youtube_transcript_file = f"youtube_{video_id}.txt"
			local_path = os.path.join(structure.txt, youtube_transcript_file)
			if os.path.exists(local_path):
				st.write(f"File {local_path} already exists")
				return

			transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
			if len(transcript):
				all_text = ''
				for seg in transcript:
					all_text += seg['text'] + ' '
				
				with open(local_path, 'w') as file:
					file.write(all_text)
				st.write(f"File downloaded and saved as {local_path}")				
				
			else:
				st.write("No transcript found")
				return
			
		except Exception as e:
			st.write(f"Failed to process url. Error: {type(e)} {e}")
			return
    
	elif ext in ['.mp3', '.wav', '.opus']:
		with st.spinner(f"Downloading {url}"):
			download_file(url, structure.audio)

	elif ext in ['.txt']:
		with st.spinner(f"Downloading {url}"):
			download_file(url, structure.txt)

	else:
		st.write("Not supported URL")
		return

def add_qa(question, answer, retrieved_docs):
	qa = ''
	if "I don't know" not in answer and retrieved_docs:
		# get file names
		retrieved_docs = [doc.metadata['source']for doc in retrieved_docs]
		
		# Extract file base names
		file_names = [doc.split("/")[-1].split(".")[0] for doc in retrieved_docs]

		# Manually count occurrences
		file_counts = {}
		for name in file_names:
			file_counts[name] = file_counts.get(name, 0) + 1

		# Create a formatted string in descending order
		source_docs = ", ".join([f"{name}({count})" for name, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)])
		qa = f"**{question}** \n\n {answer}\n\n **source**: {source_docs}\n"

	else:
		qa = f"**{question}** \n\n {answer}\n"

	ss.qas.insert(0, qa)
	show_qas()

def show_qas():
    st.subheader('Question & Answer Log', divider='rainbow') 
    for qa in ss.qas:
        st.markdown(qa)
        st.markdown('---')

def make_embedder():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def delete_all_files(directory_path):
    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def ui_load_file():
	global curent_llm

	st.write('## Upload your files, build DB and ask question')
	#disabled = not ss.get('user') or (not ss.get('api_key') and not ss.get('community_pct',0))
	t1,t2,t3 = st.tabs(['General','load from local','load from url'])

	with t1:
		st.write(f"## 2. build QA DB {ss.loaded}")
		if st.button('Build DB', disabled=not ss.loaded, use_container_width=True):
			now = datetime.now()
			current_time = now.strftime("%H:%M:%S.%f")
			
			documents = []
			documents += extract_text_from_pdf()
			documents += extract_text_from_txt()
			#st.write(f"len(documents)={len(documents)} of type {type(documents[0])} {type(documents[-1])}")

			if len(documents):
				with st.spinner(f"Building DB {current_time}"):
					text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
					texts = text_splitter.split_documents(documents)

					hf = make_embedder()
					db = Chroma.from_documents(texts, hf)
					ss['db'] = db

					if curent_llm is None and ss.get('api_key'):
						set_gf_api_key()
					else:
						st.error("API key not set")

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

					# https://python.langchain.com/docs/use_cases/question_answering/
					ss['chain'] = RetrievalQA.from_chain_type(llm=curent_llm,
													chain_type="stuff",
                                               		retriever=db.as_retriever(search_kwargs={"k": 6}),
                                            		input_key="question",
                                            		chain_type_kwargs=chain_type_kwargs)
					showtime(f"Chain build {type(ss['chain'])}")

					st.write('**building**', current_time)
					st.write('**done**')
			else:
				st.error("no text was provided to build the DB")

		# --------------------------------
		st.write('## 3. Ask questions')
		disabled = False
		question = t1.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)
		if st.button('Get answer', disabled=disabled, type='primary', use_container_width=True):
			with st.spinner('preparing answer'):
				# get source documents list
				retriever = ss.db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
				retrieved_docs = retriever.get_relevant_documents(question)

				# get asnwer and print all QA
				chain = ss['chain']
				add_qa(question, chain.run(question), retrieved_docs)

	with t2:
		c1, c2 = st.columns([2, 1])
		if c1.button('Refresh files list', use_container_width=True):
			pass

		if c2.button('Delete all files and db -- NOT REVERSIBLE !! --', use_container_width=True, type="primary"):
			del ss.chain
			ss.chain = None

			del ss.db
			ss.db = None

			del ss.qas
			ss.qas = []

			delete_all_files(structure.pdf)
			delete_all_files(structure.audio)
			delete_all_files(structure.txt)


		uploaded_file = st.file_uploader('audio, pdf and txt file',
                                   type=['pdf', 'wav', 'mp3', 'txt'],
                                   key='load from local')
		if uploaded_file is not None:
			file_save(uploaded_file)

		# List of dicts with file info
		directory = Path(structure.pdf)
		pdf_files = []
		for path in directory.glob('*.pdf'):
			info = {
				'filename': path.name,
				'size': path.stat().st_size
			}
			pdf_files.append(info)

		# Create dataframe  
		pdf_df = pd.DataFrame(pdf_files)

		# Display dataframe
		st.header('PDF Files')
		st.table(pdf_df)

		if not len(pdf_df.index) == 0:
			ss['loaded'] = True

		directory = Path(structure.audio)
		audio_files = []
		wav_files = list(directory.glob('*.wav')) 
		mp3_files = list(directory.glob('*.mp3'))

		all_audio_files = wav_files + mp3_files
		for path in all_audio_files:
			info = {
				'filename': path.name,
				'size': path.stat().st_size
			}
			audio_files.append(info)

		# Create dataframe  
		audio_df = pd.DataFrame(audio_files)

		# Display dataframe
		st.header('audio Files')
		st.table(audio_df)

		# List of dicts with file info
		directory = Path(structure.txt)
		txt_files = []
		for path in directory.glob('*.txt'):
			info = {
				'filename': path.name,
				'size': path.stat().st_size
			}
			txt_files.append(info)

		# Create dataframe
		txt_df = pd.DataFrame(txt_files)

		# Display dataframe
		st.header('Text Files')
		st.table(txt_df)

		# source: https://github.com/huggingface/distil-whisper#long-form-transcription
		if not audio_df.empty and ss.transcriber is None:
			if ss.pipe is None:
				with st.spinner('loading model'):

					device = "cuda:0" if torch.cuda.is_available() else "cpu"
					torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

					model_id = "distil-whisper/distil-medium.en" #distil-whisper/distil-large-v2"

					model = AutoModelForSpeechSeq2Seq.from_pretrained(
						model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
					)
					model.to(device)

					processor = AutoProcessor.from_pretrained(model_id)

					ss.pipe = pipeline(
							"automatic-speech-recognition",
							model=model,
							tokenizer=processor.tokenizer,
							feature_extractor=processor.feature_extractor,
							max_new_tokens=128, # controls the maximum number of generated tokens per-chunk.
							chunk_length_s=15,  # To enable chunking, pass the chunk_length_s parameter to the pipeline. For Distil-Whisper, a chunk length of 15-seconds is optimal.
							batch_size=2,
							torch_dtype=torch_dtype,
							device=device,
						)

		if st.button('Transcribe', disabled=audio_df.empty, type='primary', use_container_width=True):
			transcibe_audio()

		if not audio_df.empty or not txt_df.empty:
			ss.loaded = True
			ss.run_return = True
			time.sleep(1)

	with t3:
		text = st.text_area('Enter url based sources', key='load from url', height=100, placeholder='Enter url here', help='', label_visibility="collapsed", disabled=False)

		disabled = False
		if st.button('get files from URL list', disabled=disabled, type='primary', use_container_width=True):
			st.write('**geting**')

			if text:
				for url in text.splitlines():
					# Process each line
					#st.write(url)
					process_url(url)

			st.write('**done**')


def ui_llm():
	st.write('## 2. LLM model')
	models = ['declare-lab/flan-alpaca-large']
	st.selectbox('llm model', models, key='llm_name', on_change=set_gf_api_key,
	             disabled=not ss.get('api_key'), label_visibility="collapsed")


def ui_stt():
	st.write('## 3. STT model')
	models = ['distil-whisper/distil-medium.en']
	st.selectbox('stt model', models, key='stt_name', on_change=set_gf_api_key,
	             disabled=not ss.get('api_key'), label_visibility="collapsed")



#------------------
# ---- M A I N ----

# LAYOUT
# sidebar GUI window
with st.sidebar:
	ui_info()
	ui_spacer(2)
	
	#This code will call the set_gf_api_key function whenever the user changes the value of the huggingfacehub_api_token text input field.
	#The api_key parameter will contain the new value of the text input field.
	st.write('## 1. Enter your huggingface API key')
	st.text_input('huggingfacehub_api_token', type='password', key='api_key', on_change=set_gf_api_key, label_visibility="collapsed")
	ui_spacer(1)

	ui_llm()
	ui_spacer(1)

	ui_stt()


# main GUI window
ui_load_file()
ui_debug()



import streamlit as st
import os
import shutil
from pathlib import Path
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# source: https://github.com/huggingface/distil-whisper
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
	curent_llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})

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

    Multimodal Question Answering system supporting urls, pdf, and audio files.

	""")

# def index_pdf_file():
# 	st.write("In index_pdf_file")
# 	if ss['pdf_file']:
# 		st.write(f"In index_pdf_file {ss['pdf_file'].name}")
# 		ss['filename'] = ss['pdf_file'].name
# 		if ss['filename'] != ss.get('fielname_done'): # UGLY
# 			with st.spinner(f'indexing {ss["filename"]}'):
# 				index = model.index_file(ss['pdf_file'], ss['filename'], fix_text=ss['fix_text'], frag_size=ss['frag_size'], cache=ss['cache'])
# 				ss['index'] = index
# 				#debug_index()
# 				ss['filename_done'] = ss['filename'] # UGLY


#def ui_buildDB():
#	st.write(f"## 2. build QA DB {ss.loaded}")
#	disabled = ss.loaded
#	if st.button('Build DB', disabled=disabled, type='primary', use_container_width=True):
#		st.write('**building**')
#		st.write('**done**')


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
	loaders = []
	directory = Path(structure.pdf)
	for path in directory.glob('*.pdf'):
		st.write(f"extracting {path}")

	loaders = [UnstructuredPDFLoader(os.path.join(directory, fn)) for fn in os.listdir(directory)]
	return loaders

def extract_text_from_txt():
	loaders = []
	directory = Path(structure.txt)
	for text_file in directory.glob('*.txt'):
		st.write(f"adding {text_file}")

	loaders = [TextLoader(os.path.join(directory, fn)) for fn in os.listdir(directory)]
	return loaders

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


def download_audio(url, local_directory):
	try:
		# Extract the filename from the URL
		filename = os.path.basename(url)
		local_path = os.path.join(local_directory, filename)
		if os.path.exists(local_path):
			st.write(f"Audio file {local_path} already exists")
			return

		response = requests.get(url)
		if response.status_code == 200:

			with open(local_path, 'wb') as file:
				file.write(response.content)
			st.write(f"Audio file downloaded and saved as {local_path}")
			
		else:
			st.write(f"Failed to download audio. Status code: {response.status_code}")

	except Exception as e:
		st.write(f"An error occurred: {str(e)}")


def process_url(url):
    
	# Check if valid URL 
	if not re.match(r"https?://", url):
		print("Not a valid URL")  
		return
        
	parsed_url = urlparse(url)
	ext = os.path.splitext(parsed_url.path)[1]
    
	if parsed_url.hostname == 'youtube.com':
		st.write(f"{url} is a youtube url")
    
	elif ext in ['.mp3', '.wav', '.opus']:
                  
		# Download file  
		with st.spinner(f"Downloading {url}"):
			download_audio(url, structure.audio)

		# audio_data = requests.get(url).content 

		# # Save in wav directory
		# filename = structure.audio + '/' + os.path.basename(url)
		# with open(filename, 'wb') as f:
		# 	f.write(audio_data)

		# print(f"Saved audio file to {filename}")
	
	else:
		st.write("Not supported URL")
		return

def add_qa(question, answer):
    qa = f"**{question}** \n\n {answer}\n\n"
    ss.qas.insert(0, qa)
    show_qas()

def show_qas():
    st.subheader('Question & Answer Log', divider='rainbow') 
    for qa in ss.qas:
        st.markdown(qa)
        st.markdown('---')

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

			loaders = []
			loaders = extract_text_from_pdf()
			loaders += extract_text_from_txt()
			#st.write(loaders)

			if len(loaders):
				with st.spinner(f"Building DB {current_time}"):
					vectorstoreIndex = VectorstoreIndexCreator(
						embedding=HuggingFaceEmbeddings(),
						text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

					if curent_llm is None and ss.get('api_key'):
						set_gf_api_key()
					else:
						st.error("API key not set")

					# https://python.langchain.com/docs/use_cases/question_answering/
					ss['chain'] = RetrievalQA.from_chain_type(llm=curent_llm,
													chain_type="stuff",
													retriever=vectorstoreIndex.vectorstore.as_retriever(search_kwargs={"k": 6}),
													input_key="question")
					showtime(f"Chain build {type(ss['chain'])}")

					st.write('**building**', current_time)
					st.write('**done**')
			else:
				st.error("no text was provided to build the DB")


		st.write('## 3. Ask questions')
		disabled = False
		question = t1.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)
		if st.button('get answer', disabled=disabled, type='primary', use_container_width=True):
			with st.spinner('preparing answer'):
				chain = ss['chain']
				# get asnwer and print all QA
				add_qa(question, chain.run(question))

	with t2:
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

		if not len(audio_df.index) == 0:
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


# main GUI window
ui_load_file()
ui_debug()

#
#Learn and remember:
#
# .........
# 1. Session state
#st.session_state is a way to store state across reruns in Streamlit.
#
#When a Streamlit app reruns (for example when interacting with a widget), normally all state is lost. 
#st.session_state allows you to store values across reruns so they persist.
#
#    import streamlit as st
#
#    if 'count' not in st.session_state:
#        st.session_state.count = 0
#
#    st.session_state.count += 1
#
#    st.write(st.session_state.count)
#
#This will increment and display a counter that persists across reruns.
#
# .........
# 2.
#

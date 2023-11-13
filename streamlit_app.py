import streamlit as st
import os
import shutil
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
from datetime import datetime
import time


app_name = "streamlit RAG"

st.set_page_config(layout='centered', page_title=f'{app_name}')
ss = st.session_state

if 'debug' not in ss: ss['debug'] = {}
if 'loaded' not in ss: ss['loaded'] = False
if 'run_return' not in ss: ss['run_return'] = False

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

    Description: TBD
	""")

def index_pdf_file():
	st.write("In index_pdf_file")
	if ss['pdf_file']:
		st.write(f"In index_pdf_file {ss['pdf_file'].name}")
		ss['filename'] = ss['pdf_file'].name
		if ss['filename'] != ss.get('fielname_done'): # UGLY
			with st.spinner(f'indexing {ss["filename"]}'):
				index = model.index_file(ss['pdf_file'], ss['filename'], fix_text=ss['fix_text'], frag_size=ss['frag_size'], cache=ss['cache'])
				ss['index'] = index
				#debug_index()
				ss['filename_done'] = ss['filename'] # UGLY


def ui_buildDB():
	st.write(f"## 2. build QA DB {ss.loaded}")
	disabled = ss.loaded
	if st.button('Build DB', disabled=disabled, type='primary', use_container_width=True):
		st.write('**building**')
		st.write('**done**')


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
	directory = Path(structure.pdf)
	for path in directory.glob('*.pdf'):
		st.write(f"extracting {path}")

def	transcibe_audio():
	directory = Path(structure.audio)
	wav_files = list(directory.glob('*.wav')) 
	mp3_files = list(directory.glob('*.mp3'))

	all_audio_files = wav_files + mp3_files
	for path in all_audio_files:
		st.write(f"transcribing {path}")


def ui_load_file():
	st.write('## Upload your files, build DB and ask question')
	#disabled = not ss.get('user') or (not ss.get('api_key') and not ss.get('community_pct',0))
	t1,t2,t3 = st.tabs(['General','load from local','load from url'])

	t1.write(f"## 2. build QA DB {ss.loaded}")
	#disabled = ss.loaded
	if t1.button('Build DB', disabled=not ss.loaded, use_container_width=True):
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S.%f")

		extract_text_from_pdf()
		transcibe_audio()

		st.write('**building**', current_time)
		st.write('**done**')

	t1.write('## 3. Ask questions')
	disabled = False
	if t1.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled):
		pass

	#with t1:
	#		#ui_buildDB()
	#	ui_question()

	with t2:
		uploaded_file = st.file_uploader('pdf file', type=['pdf', 'wav', 'mp3'], key='load from local')

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
				for line in text.splitlines():
					# Process each line
					st.write(line)

			st.write('**done**')

		# filenames = ['']
		# if ss.get('storage'):
		# 	filenames += ss['storage'].list()
		# def on_change():
		# 	name = ss['selected_file']
		# 	if name and ss.get('storage'):
		# 		with ss['spin_select_file']:
		# 			with st.spinner('loading index'):
		# 				t0 = now()
		# 				index = ss['storage'].get(name)
		# 				ss['debug']['storage_get_time'] = now()-t0
		# 		ss['filename'] = name # XXX
		# 		ss['index'] = index
		# 		#debug_index()
		# 	else:
		# 		#ss['index'] = {}
		# 		pass

		# st.selectbox('select file', filenames, on_change=on_change, key='selected_file', label_visibility="collapsed", disabled=False)
		# #b_delete()
		# ss['spin_select_file'] = st.empty()

# ---- M A I N ----

#st.write(structure.pdf)
#st.write(structure.audio_txt)

# LAYOUT
# sidebar GUI window
with st.sidebar:
	ui_info()
	ui_spacer(2)

# main GUI window
ui_load_file()


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

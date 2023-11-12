import streamlit as st
import os
import shutil
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
from datetime import time, datetime


app_name = "streamlit RAG"

st.set_page_config(layout='centered', page_title=f'{app_name}')
ss = st.session_state
if 'debug' not in ss: ss['debug'] = {}


def showtime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")
    st.write("Current Time =", current_time)


class DirectoryStructure:

    def __init__(self):
       self.create_structure()

    def create_structure(self):
        # List of directories to create
        dirs = [
            "pdf",
            os.path.join("pdf", "txt"),
            "audio",
            os.path.join("audio", "txt")
        ]

        # Loop through list and create directories
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

    @property
    def pdf(self):
        return os.path.realpath("pdf")
    
    @property
    def pdf_txt(self):
        return os.path.realpath(os.path.join("pdf", "txt"))

    @property
    def audio(self):
        return os.path.realpath("audio")

    @property
    def audio_txt(self):
        return os.path.realpath(os.path.join("audio", "txt"))


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

def pdf_save(uploaded_file, path=structure.pdf):
    
	# Get file name and type
	file_name = Path(uploaded_file.name).stem
	file_ext = Path(uploaded_file.name).suffix

	# Save file to local file system
	local_file = os.path.join(path, f'{file_name}{file_ext}')
	with open(local_file, 'wb') as out_file:
		shutil.copyfileobj(uploaded_file, out_file)

	#st.success(f'Saved PDF to {local_file}')

	# remember
	# Save uploaded file  
	#with open(f"{uploaded_file.name}", "wb") as f: 
	#	f.write(uploaded_file.getbuffer())


def ui_pdf_file():
	st.write('## 1. Upload or select your PDF file')
	disabled = not ss.get('user') or (not ss.get('api_key') and not ss.get('community_pct',0))
	t1,t2,t3 = st.tabs(['General','load from local','update load list'])
	with t1:
		ui_buildDB()
		ui_question()

	with t2:
		uploaded_file = st.file_uploader('pdf file', type='pdf', key='pdf_file')
		#b_save()

		if uploaded_file is not None:
			#st.write(f"{uploaded_file.name=}")
			pdf_save(uploaded_file)

		# Specify directory 
		directory = Path(structure.pdf)

		# List of dicts with file info
		pdf_files = []
		for path in directory.glob('*.pdf'):
			info = {
				'filename': path.name,
				'size': path.stat().st_size
			}
			pdf_files.append(info)

		# Create dataframe  
		df = pd.DataFrame(pdf_files)

		# Display dataframe
		st.header('PDF Files')
		st.table(df)


	with t3:
		filenames = ['']
		if ss.get('storage'):
			filenames += ss['storage'].list()
		def on_change():
			name = ss['selected_file']
			if name and ss.get('storage'):
				with ss['spin_select_file']:
					with st.spinner('loading index'):
						t0 = now()
						index = ss['storage'].get(name)
						ss['debug']['storage_get_time'] = now()-t0
				ss['filename'] = name # XXX
				ss['index'] = index
				#debug_index()
			else:
				#ss['index'] = {}
				pass

		st.selectbox('select file', filenames, on_change=on_change, key='selected_file', label_visibility="collapsed", disabled=disabled)
		#b_delete()
		ss['spin_select_file'] = st.empty()


def ui_buildDB():
	st.write('## 2. build QA DB')
	disabled = False
	if st.button('Build DB', disabled=disabled, type='primary', use_container_width=True):
		st.write('**building**')
		st.write('**done**')


def ui_question():
	st.write('## 3. Ask questions')
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)

# ---- M A I N ----

st.write(structure.pdf)
st.write(structure.audio_txt)

# LAYOUT
# sidebar GUI window
with st.sidebar:
	ui_info()
	ui_spacer(2)

# main GUI window
ui_pdf_file()


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

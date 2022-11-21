# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 00:30:40 2022

@author: Suklesh
"""
import streamlit as st
import pandas as pd
import numpy as np
from pyresparser import ResumeParser
import os
import spacy
#import spacy.load.'en_core_web_sm'
import  docx2txt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import base64,random
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import docx

#pip install New

path = r'F:\ExcelR\Project 2 - NLP\Resumes-20220812T140008Z-001\Resumes\New folder'

st.set_page_config(page_title='Resume Classifier',page_icon='F:\\ExcelR\\Project 2 - NLP\\deploy.jpg')

images_source='F:\\ExcelR\\Project 2 - NLP\\deploy.jpg'

def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(images_source);
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg(images_source)

st.sidebar.image('F:\\ExcelR\\Project 2 - NLP\\titleimg.png')
st.title("Resume Classification")
uploaded_files= st.sidebar.file_uploader("Upload Resume", type=None, accept_multiple_files=True ,key=None, help=None,
                         on_change=None, args=None, kwargs=None, disabled=False)

res_list=[]
file_name=[]
complete_path=[]
skills_list=[]

def readingtextdoc(filename):
    doc = docx.Document(filename)
    completedText = []
    for paragraph in doc.paragraphs:
        completedText.append(paragraph.text)
    return '\n'.join(completedText)

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    
# Fetching Records.
for uploaded_file in uploaded_files:
    with st.spinner('Please wait while file is loading...'):
        bytes_data = uploaded_file.read()
        #st.write("filename:", uploaded_file.name)
        filename=uploaded_file.name
        pfinal = os.path.join(paths, filename)
        #st.write(pfinal)
        #count1=count1+1
        complete_path.append(pfinal)
        txt = docx2txt.process(pfinal)
        if txt:
            data= txt.replace('\n',' ').replace('\t','   ')
        #return None
        res_list.append(data)
        file_name.append(filename)
        skills = ResumeParser(pfinal).get_extracted_data()
        skills_list.append(skills['skills'])
        # with open(pfinal, "wb") as f:
        #     f.write(pdf_file.getbuffer())
        #     show_pdf(pfinal)

if len(uploaded_files)>0:
    with st.spinner('Loading.....'):
        new_data = pd.DataFrame({"Resume":res_list})
        if len(uploaded_files) == 1:
            st.header(filename )
            st.subheader(readingtextdoc(pfinal))
            st.header('Skills Set - '+filename)
        else:
            st.header('Resume')
            st.write(new_data)
            st.header('Skills Set')
            
        k=0
        j=1
        for i in skills_list:
            if len(uploaded_files) != 1:
                st.subheader(str(j) +' . ' +((file_name)[k])+"")
            else:
                pass
            st.success(i)
            k=k+1
            j=j+1

#Cleaning the data
if len(uploaded_files)>0:
    def clean_text(kit):
        kit=str(kit).lower()
        kit=re.sub(r"@\S+",r' ',kit)
        kit=re.sub('\[.*?\]',' ',kit)
        kit=re.sub("\s+",' ',kit)
        kit=re.sub("\n",' ',kit)
        re.sub(r'\s+[a-zA-Z]\s+', ' ', kit)
        kit=re.sub('[^a-zA-Z]',' ',kit)
        kit=re.sub('[''""]',' ',kit)
        kit=re.sub(r'\s+', ' ', kit, flags=re.I)
        letters=re.sub('[%s]'% re.escape(string.punctuation),'',kit)
        return letters
    clean1=lambda x:clean_text(x)
    new_data['clean_resume']=new_data['clean_resume'].apply(clean1)

#Stop Words Removal    
stop_words=stopwords.words('english')
type(stop_words)
stop_words.extend(["n","x","xe","xa","xae","xc","using","mp","b","etc"])

new_data['clean_resume']=new_data['clean_resume'].apply(lambda x:' '.join([kit for kit in x.split(' ') if kit not in stop_words]))

if len(uploaded_files)>0:
    with st.spinner('Loading.....'):
        tfid_vector = pickle.load(open('transform.pkl','rb'))
        model = pickle.load(open('nlp_model.pkl','rb'))
        vector_input = tfid_vector.transform(new_data['Clean_Resume'])
        #st.write(vector_input.shape)
        result = model.predict(vector_input)
        st.header('Job Designation based on Skills Set')
        k=0
        count_ReactJs=0
        count_Peoplesoft=0
        count_SQL=0
        count_Workday=0
        category_values=[]
        
        for i in result:
            #for j in ((file_name)[k]):
            #    st.write(j)
            #st.write(i)
            if i == 0:
                st.info("Resume looks to be from PeopleSoft - "+ ((file_name)[k]) +"")
                count_Peoplesoft=count_Peoplesoft+1
                category_value=1
            elif i == 1:
                st.info("Resume looks to be a ReactJS Developer - "+ ((file_name)[k]) +"")
                count_ReactJs =count_ReactJs + 1
                category_value=1
            elif i == 2:
                st.info("Resume looks to be an SQL Developer - "+ ((file_name)[k]) +"")
                count_SQL = count_SQL + 1
                category_value=2
            elif i == 3:
                st.info("Resume looks to be from Workday - "+ ((file_name)[k]) +"")
                count_Workday = count_Workday + 1
                category_value=3
            else:
                pass
            k=k+1;
            category_values.append(category_value)
            
        dict = {'Job': ['Peoplesoft Resume','React JS Developer', 'SQL Developer','Workday Resume'],
                'Values': [count_Peoplesoft,count_ReactJs, count_SQL, count_Workday]}
        dict=pd.DataFrame(dict)

        if st.sidebar.button('Click here to check the Bar Chart'):
            mylabels = ['Peoplesoft Resume','React JS Developer', 'SQL Developer','Workday Resume']
            fig = px.bar(dict, x='Job', y='Values')
            fig.show()

        if st.sidebar.button('Click here to check the Pie Chart'):
            mylabels = ['Peoplesoft Resume','React JS Developer', 'SQL Developer','Workday Resume']
            #     # plt.pie(pie_data['Category_counts'].value_counts(),labels=list[mylabels])
            #     # plt.title('Distribution of Job title in Resume',fontsize=20)
            #     data = clean_data['Category'].value_counts()
            fig = px.pie(values=dict['Values'], names=dict['Job'])
            fig.show()
    
    
    
    
    
    
    
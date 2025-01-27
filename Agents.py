

import asyncio
from browser_use import Agent
import json
import requests
from airtable import airtable
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import pdfplumber

import gradio as gr
import json
import os
from dotenv import load_dotenv
import yaml

async def run_search(job_link, username, password) -> str:
 

    task_str_eachjob = f"""Perform the following steps.
    1. Go to  {job_link} 
    2. If you are taken to the 'Sign In' page.Use the following credentials to login. 'Email or phone': {username}  'Password': {password}
    3. You will see the job details page. Click 'See more'. This will open the details page for the job. Read the job details and extract the following information from job details page 
         a. Get the job title and company name and location. 
         b. Get the Job Description from the details page. Look for sections titled 'What we do' or 'About the Job', or similar.
         c. Look for sections titled 'What you'll do' or 'What you can expect in your role' or 'The impact you will make', or similar, and get the text from that section 
         d. Look for sections titled 'What you'll bring' or 'You are', or 'Attributes for success' or 'Nice to Have's', or similar. 
    4. Return the result as a dictionary with the following keys: 'job_title', 'company_name', 'location', 'job_description', 'job_requirements', 'skills_required'
    5. Return a formatted json dictonary that can be converted to json object
"""
       

    agent = Agent(
        task=task_str_eachjob,
        llm=llm
        )
    
    search_agent = Agent(task=task_str_eachjob, llm=llm)
    result = await search_agent.run()
   
    return result


def add_new_record(job_profile):
   new_fields = {
    "job_profile": job_profile["job_title"] + " at " + job_profile["company_name"] + " in " + job_profile["location"],
    "job_profile_description": job_profile["job_description"] + "\n\n" + job_profile["job_requirements"] + "\n\n" + job_profile["skills_required"],
    "tailored_resume": "",
    "status": "new"
   }
   data = {
    'fields': new_fields
    }
   url = f"{AIRTABLE_URL}/resumes"
   headers = {
      'Authorization': f'Bearer {AIRTABLE_TOKEN}',
      'Content-Type': 'application/json'
    }
   response = requests.request("POST", url, headers=headers, json=data)
   return response.json()

def get_all_records():
    url = f"{AIRTABLE_URL}/resumes"
    result = at.get('resumes')
    all_records = []
    for record in result["records"]:
        fields = record['fields']
        all_records.append(record)
    return all_records



def get_new_records():
    url = f"{AIRTABLE_URL}/resumes"
    result = at.get('resumes')
    new_records = []
    for record in result["records"]:
        fields = record['fields']
        if(fields.get('status')=="new"):
            new_records.append(record)
    return new_records


def get_resume_generated_records():
    url = f"{AIRTABLE_URL}/resumes"
    result = at.get('resumes')
    new_records = []
    for record in result["records"]:
        fields = record['fields']
        if(fields.get('status')=="resume_generated"):
            new_records.append(record)
    return new_records

def get_job_description(job_filename):
  job_description=""
  resume_text=""
  with open(job_filename, 'r') as file:
      job_description = file.read()
  return job_description

def get_resume_text(resume_filename):
  resume_text=""
  if(resume_filename.endswith('.pdf')):
      with pdfplumber.open(resume_filename) as pdf:
       resume_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
  else:
      with open(resume_filename, 'r') as file:
        resume_text = file.read()
  return resume_text

async def start_process(resume_filename, job_file):
    response = await parse_job_profiles(job_file)
    response  = generate_tailored_resumes_for_new_records()
    return response

def generate_tailored_resume( resume_filename,job_description, creativity_slider,tailoring_prompt):
  final_prompt =  tailoring_prompt + f"""
  {{job_description}}
  {{resume_text}}


  """
  resume_text = get_resume_text(resume_filename)
  final_prompt = final_prompt.format(job_description=job_description, resume_text=resume_text)
  gemini_llm.temperature = creativity_slider
  response = gemini_llm.invoke(final_prompt).content
  return response

credentials = None
gemini_key = None
openai_key = None
username = None
password = None

def load_secrets():
    global credentials, gemini_key, openai_key, username, password
    credentials = yaml.load(open('credentials.yml'),  Loader=yaml.FullLoader)
    username = credentials['username']
    password = credentials['password']
    gemini_key = credentials['gemini_key']
    openai_key = credentials['openai_key']
# access values from dictionary


async def parse_job_profiles(job_profiles_filename):
    jobs = []
    with open(job_profiles_filename) as file:
        for line in file:
            job_link = line.rstrip()
            result = await run_search(job_link, username, password)
            jobs.append(result.final_result())  
    for job in jobs:
     job_description_json = json.loads(job)
     response = add_new_record(job_description_json)

def generate_tailored_resumes_for_new_records():
    new_records = get_new_records()
    for record in new_records:
        fields = record['fields']
        tailored_resume =  generate_tailored_resume("resume.pdf",fields['job_profile']+ "\n" + fields['job_profile_description'],0.3,prompt)
        fields['tailored_resume'] = tailored_resume
        fields['status'] = "resume_generated"
        response =at.update('resumes', record['id'], fields)
        


load_secrets()
job_profiles_filename = "jobprofiles.txt"
AIRTABLE_TOKEN='patRXkQM1aZAgiGPI.e1000671c81e9a3f386a90b6aed5d42e477b54b53675955f2754a927ab20c478'
AIRTABLE_BASE_ID='appyIMGKqO0Elv6VM'
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
at = airtable.Airtable(AIRTABLE_BASE_ID, AIRTABLE_TOKEN)



creativity_level = 0.3
gemini_model_name = "gemini-pro"
gemini_key= gemini_key 
gemini_llm = ChatGoogleGenerativeAI(model=gemini_model_name, google_api_key=gemini_key, temperature=creativity_level)
model_name = "gpt-4o-mini"


llm = ChatOpenAI(
    model_name=model_name,
    temperature=0.0,
    timeout=100, # Increase for complex tasks
    api_key= openai_key 
)

prompt=f"""

You are a very competent resume analyzer that generates resumes tailored to specific job profiles.

You have been provided with a job description and a resume. Your job is to generate a tailored resume for the user.

Extract the following from the job description.
Job Responsibilities: Outline the primary tasks and expectations for the role.This can be specified in different ways like "What you will do, Your daily tasks, Your responsibilities, Your duties etc. or something similarly worded"
Job Qualifications: Highlight the required skills, experience, and knowledge.

Tailor all relevant sections the original resume. As an example you can tailor the following sections in the resume while keeping other sections intact. But if the resume has other relevant sections like projects etc. that can be tailored
please tailor that too.

Skills Section:
Retain the original formatting and structure of the resume’s skills section.
Add keywords and skills mentioned in the job description that are critical for the role.
Ensure the length of this section remains consistent with the original resume.

Summary Section:
Rewrite the summary to highlight relevant experiences and skills from the resume that align with the job responsibilities and qualifications.
Use industry buzzwords and metrics where applicable, keeping the tone professional and concise.

Professional Experience Section:
- Enhance each job entry by rewording and restructuring responsibilities to emphasize alignment with the job responsibilities and job qualifications.-
- Integrate key performance indicators (KPIs), metrics, and achievements directly relevant to the role.
- Focus on areas of overlap between the candidate's resume and the job description. Avoid addressing qualifications or responsibilities not supported by the original resume content.
- Do not add or infer skills, experiences, or achievements that are not explicitly stated or cannot be reasonably derived from the original resume.
- Use the language, tone, and keywords from the job description wherever they authentically match the candidate's experiences.
- Do not exaggerate the scope, scale, or impact of the candidate's accomplishments.
-THIS is very important - Count the number of experiences in professional section and compare it to the experiences in tailored resume. If the count in tailored resume is less than the ones in original resume, add the missing ones.

**Output:**
Generate a full resume for the user. The tailored section would have all sections from the original resume
- Add a new line between each section
- Make the section headers in CAPS LOCK. Do not add any extra characters like ****
- Add a bullet point for each job responsibilities for each section
- Remove any surrounding characters like `***` or other markers.
- Avoid any extra commentary or introductory/explanatory text. Present only the tailored resume content.
- THIS is very important - Count the number of experiences in professional section and compare it to the experiences in tailored resume. If the count in tailored resume is less than the ones in original resume, do the tailoring for
the remamning experiences and add it to the tailored resume.
- Do not include job profile in the output content
"""
import pandas as pd

# df = pd.DataFrame(columns=['job_profile', 'job_profile_description', 'tailored_resume','status'])
# df.style.hide_columns(['job_profile_description', 'tailored_resume'])

def get_dashboard_data()->pd.DataFrame:
    job_status = {}
    df = pd.DataFrame(columns=['job_profile', 'job_profile_description', 'tailored_resume','status'])
    records = get_all_records()
    # new_records = get_resume_generated_records()
    for record in records:
        fields = record['fields']

        df.loc[len(df)] = fields
    return df

def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
        print("index", evt.index)
        print("value", evt.value)
        print("row_value", evt.row_value)
        return [evt.row_value[1], evt.row_value[2]]
#*********************************************/
    
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Resume Tailoring")
    with gr.Row():
     with gr.Column(scale=1):
        pdf_input = gr.File(label="Upload resume profile", file_types=[".pdf",".docx"], visible=True)
        job_file = gr.File(label="Upload job links", file_types=["text"], visible=True)
        start_process_button= gr.Button("Start Processing", visible=True)
        refresh_button= gr.Button("Refresh", visible=True)
     with gr.Column(scale=5):
        with gr.Row():
         df = gr.DataFrame(get_dashboard_data, visible=True, interactive=True)
        with gr.Row():
          job_profile_description = gr.Textbox(label="job_profile_description")
          tailored_resume = gr.Textbox(label="Tailored Resume")
    # with gr.Column(scale=3):
    # with gr.Row():
    #        creativity_slider = gr.Slider(minimum=0, maximum=1, value=creativity_level, label="Creativity Level", interactive=True)
    # with gr.Row():
    #       tailoring_prompt = gr.Textbox(label="Prompt", value=prompt)
        # with gr.Row():
        #   reset_prompt_button= gr.Button("Reset Prompt", visible=True)
    start_process_button.click(
        start_process,
        inputs=[pdf_input, job_file],
       
      )
    refresh_button.click(
        get_dashboard_data,
        inputs=[],
        outputs=[df]
      )
    df.select(df_select_callback, inputs=[df], outputs=[job_profile_description, tailored_resume])
    #  resume_analysis_button.click(
    #     generate_tailored_resume,
    #     inputs=[pdf_input, job_description, creativity_slider, tailoring_prompt],
    #     outputs=[output]
    #   )
    #  reset_prompt_button.click(
    #     reset_prompt,
    #     inputs=[tailoring_prompt],
    #     outputs=[tailoring_prompt]
    #   )

      #/**********************************************
demo.launch(debug=True)
# if __name__ == "__main__":
#     asyncio.run(main())
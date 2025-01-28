import sys

import json
import requests
import pdfplumber
import gradio as gr
import os
import yaml
from dotenv import load_dotenv
from pyairtable import Api
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from pyairtable.formulas import AND, GTE, Field, match

# Load environment variables
load_dotenv()

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "resumes"

# Initialize Airtable Table
api = Api(api_key=AIRTABLE_TOKEN)
table = api.table( AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

# Load other credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

prompt=f"""

You are a very competent resume analyzer that generates resumes tailored to specific job profiles.

You have been provided with a job description and a resume. Your job is to generate a tailored resume for the user.
Job description: {{job_description}}
Resume text: {{resume_text}}
Extract the following from the job description.
Job Responsibilities: Outline the primary tasks and expectations for the role.This can be specified in different ways like "What you will do, Your daily tasks, Your responsibilities, Your duties etc. or something similarly worded"
Job Qualifications: Highlight the required skills, experience, and knowledge.

Tailor following sections of the original resume while keeping the other sections intact
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
- Do not include job profile in the output content
"""

# Initialize LLMs
creativity_level = 0.3
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=creativity_level
)
llm = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0.0, timeout=100, api_key=OPENAI_API_KEY
)
# CRUD Operations with Airtable
def add_new_record(job_profile):
    """Adds a new job profile to Airtable."""
    new_fields = {
        "job_profile": f"{job_profile['job_title']} at {job_profile['company_name']} in {job_profile['location']}",
        "job_profile_description": f"{job_profile['job_description']}\n\n{job_profile['job_requirements']}",
        "tailored_resume": "",
        "status": "new"
    }
    return table.create(new_fields)

def get_all_records():
    """Fetches all records from Airtable."""
    return table.all()

def get_new_records():
    """Fetches records with the status 'new'."""
    formula = match({"status": 'new'})
    #formula = "{status} = 'new'"
    return table.all(formula= formula)

def update_record(record_id, updated_fields):
    """Updates a specific record in Airtable."""
    return table.update(record_id, updated_fields)
# Helper Functions
def get_resume_text(resume_filename):
    """Extracts text from a resume file (PDF or plain text)."""
    resume_text = ""
    if resume_filename.endswith(".pdf"):
        with pdfplumber.open(resume_filename) as pdf:
            resume_text = " ".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    else:
        with open(resume_filename, "r") as file:
            resume_text = file.read()
    return resume_text

def generate_tailored_resume(resume_filename, job_description, tailoring_prompt):
    """Generates a tailored resume using the job description and a prompt."""
    resume_text = get_resume_text(resume_filename)
    final_prompt = tailoring_prompt.format(
        job_description=job_description, resume_text=resume_text
    )
    print(final_prompt)
    return gemini_llm.invoke(final_prompt).content

async def run_search(job_link, username, password) -> str:
    """Uses the browser agent to extract job details."""
    task_str_eachjob = f"""
    Perform the following steps:
    1. Go to {job_link}.
    2. Use credentials to log in: Email: {username}, Password: {password}.
    3. Extract the job title, company name, location, job description, and job requirements.
    4. Return the result as a JSON dictionary.
    """
    agent = Agent(task=task_str_eachjob, llm=llm)
    result = await agent.run()
    return result
async def parse_job_profiles(job_profiles_filename):
    """Parses job profiles and adds them to Airtable."""
    with open(job_profiles_filename, "r") as file:
        for line in file:
            job_link = line.strip()
            if not job_link:
                continue
            result = await run_search(job_link, USERNAME, PASSWORD)
            if result:
                job_description_json = json.loads(result.final_result())
                add_new_record(job_description_json) 

def generate_tailored_resumes_for_new_records(resume_filename,prompt):
    """Generates tailored resumes for all new job records."""
    new_records = get_new_records()
    for record in new_records:
        fields = record["fields"]
        tailored_resume = generate_tailored_resume(
            resume_filename,
            fields["job_profile"] + "\n" + fields["job_profile_description"],
            prompt
        )
        updated_fields = {
            "tailored_resume": tailored_resume,
            "status": "resume_generated"
        }
        update_record(record["id"], updated_fields)

async def start_process(resume_filename, job_file,prompt):
    """Starts the process of parsing job profiles and generating tailored resumes."""
    # await parse_job_profiles(job_file)
    generate_tailored_resumes_for_new_records(resume_filename,prompt)
    return "Process complete!"
# Gradio Dashboard
def get_dashboard_data() -> pd.DataFrame:
    """Fetches data for the Gradio dashboard."""
    df = pd.DataFrame(columns=["job_profile", "job_profile_description", "tailored_resume", "status"])
    records = get_all_records()
    for record in records:
        fields = record["fields"]
        df.loc[len(df)] = fields
    return df

def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
        print("index", evt.index)
        print("value", evt.value)
        print("row_value", evt.row_value)
        return [evt.row_value[1], evt.row_value[2]]

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Resume Tailoring Dashboard")
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload Resume", file_types=[".pdf"])
            job_file = gr.File(label="Upload Job Links", file_types=["text"])
            start_button = gr.Button("Start Processing")
            refresh_button = gr.Button("Refresh")
            prompt = gr.Textbox(label="Prompt", value=prompt,interactive=True)
        with gr.Column(scale=5):
            df = gr.DataFrame(get_dashboard_data, visible=True, interactive=True)
            job_description = gr.Textbox(label="Job Description")
            tailored_resume = gr.Textbox(label="Tailored Resume")

    async def process_files(resume_file, job_links_file,prompt):
        return await start_process(resume_file, job_links_file,prompt)

    start_button.click(process_files, inputs=[pdf_input, job_file, prompt], outputs=[])
    refresh_button.click(get_dashboard_data, inputs=[], outputs=[df])
    df.select(df_select_callback, inputs=[df], outputs=[job_description, tailored_resume])
demo.launch(debug=True)


# Prompt for Tailoring Resumes






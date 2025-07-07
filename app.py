from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import pipeline
import random
from utils import google_translate
from Questgen import main
from Questgen.mcq.mcq import tokenize_sentences, get_options
from LlamaModel import explain_concept

example = {
	"input_text": "In recent years, technology has profoundly transformed the landscape of education. Digital tools such as computers, tablets, and the internet have made information more accessible than ever before, enabling students to learn beyond traditional classroom boundaries. Online platforms and educational software facilitate personalized learning experiences, catering to individual student needs and learning styles. Moreover, technology promotes collaboration among students and teachers through virtual classrooms and communication tools, fostering a more interactive and engaging learning environment. However, despite these benefits, challenges such as the digital divide and potential distractions from devices must be addressed to ensure equitable and effective education for all.",
 	"max_questions": "1"
}

questGenModel = main.QGen()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

class MCQRequest(BaseModel):
    input_text: str
    max_questions: int
@app.post("/generate-mcq")
async def generate_mcq(request: MCQRequest):
    text = request.input_text.replace('\n', ' ')
    translated_text = await google_translate(text, src='ko', dest='en')
    payload = {
        "input_text": translated_text,
        "max_questions": request.max_questions
    }
    output = questGenModel.predict_mcq(payload)

    questions = output["questions"]

    translated_questions = []
    for question in questions:
        texts_to_translate = [
            question["question_statement"],
            question["answer"],
            *question["options"],
            question["context"]
        ]
        translated_results = await google_translate(texts_to_translate, src='en', dest='ko')

        question_statement = translated_results[0]
        answer = translated_results[1]
        options = translated_results[2:2+len(question["options"])]
        context = translated_results[-1]

        translated_questions.append({
            "question_statement": question_statement,
            "answer": answer,
            "options": options,
            "context": context
        })

    return translated_questions

class ValidMCQCountRequest(BaseModel):
    input_text: str
@app.post("/valid_mcq_question_count")
async def valid_mcq_question_count(request: ValidMCQCountRequest):
    text = request.input_text.replace('\n', ' ')
    translated_text = await google_translate(text, src='ko', dest='en')

    valid_mcq_question_count = questGenModel.get_valid_mcq_count(translated_text)
    return {"valid_mcq_question_count": valid_mcq_question_count}

@app.get("/explain_stream")
async def explain_stream(request: Request):
    word = request.query_params.get("word")
    
    return StreamingResponse(explain_concept(word), media_type="text/event-stream")
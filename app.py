from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import pipeline
import random

example = {
	"input_text": "In recent years, technology has profoundly transformed the landscape of education. Digital tools such as computers, tablets, and the internet have made information more accessible than ever before, enabling students to learn beyond traditional classroom boundaries. Online platforms and educational software facilitate personalized learning experiences, catering to individual student needs and learning styles. Moreover, technology promotes collaboration among students and teachers through virtual classrooms and communication tools, fostering a more interactive and engaging learning environment. However, despite these benefits, challenges such as the digital divide and potential distractions from devices must be addressed to ensure equitable and effective education for all.",
 	"max_questions": "1"
}

model ={
    "qg": None,
    "summarizer": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from Questgen import main
    model["qg"] = main.QGen()
    model["summarizer"] = pipeline(
        "summarization", 
        model="lcw99/t5-base-korean-text-summary", 
        tokenizer="lcw99/t5-base-korean-text-summary") #alter: finetune model
    
    yield

app = FastAPI(lifespan=lifespan)

class MCQRequest(BaseModel):
    input_text: str
    max_questions: int = 5
@app.post("/generate-mcq")
def generate_mcq(request: MCQRequest):
    payload = {
        "input_text": request.input_text,
        "max_questions": request.max_questions
    }
    output = model["qg"].predict_mcq(payload)

    options = output["questions"][0]["options"] + output["questions"][0]["extra_options"]
    options = random.sample(options, 3)

    result = {
        "type": output["questions"][0]["question_type"],
        "question": output["questions"][0]["question_statement"],
        "answer": output["questions"][0]["answer"],
        "wrongAnswers": options
    }

    return result

class SummarizeRequest(BaseModel):
    text: str
@app.post("/summarize")
def summarize_text(request: SummarizeRequest):
    text = request.text
    summary = model["summarizer"](
        text,
        max_length=200,
        min_length=20,
        truncation=True
    )

    return {
        "summary": summary[0]["summary_text"]
    }
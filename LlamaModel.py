import lmstudio as lms

SERVER_URL = "localhost:1234"
lms.configure_default_client(SERVER_URL)
print(lms.list_loaded_models())
model = lms.llm("meta-llama-3-8b-instruct")

def explain_concept(word):
    prompt = f"Explain the concept of '{word}' in simple English. summarize the concept short."
    
    result = model.complete_stream(
        prompt, 
        config={
            "maxTokens": 100,
        }
    )
    for fragment in result:
        yield f"data: {fragment.content}\n\n"


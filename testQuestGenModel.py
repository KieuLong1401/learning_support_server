from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mock input (context and answer)
context = 'Typical operating systems include Windows, MacOS, Linux, and Android and iOS in the mobile environment, which are designed based on different kernel structures, system call interfaces, and security policies to meet various user demands. Operating System (OS) is a core software of computer systems, serving as a mediator between the user and hardware, providing efficient management and program execution environment of system resources. Memory management optimizes limited physical memory through address space division, virtual memory, and page replacement algorithms, and plays a key role in multitasking and multi-user environments.'
answer = "user"
# Highlight the answer in the context with <hl> tokens as required by the model
highlighted_context = context.replace(answer, f"<hl> {answer} <hl>")
input_text = f"generate question: {highlighted_context}"

# Tokenize and generate
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Context:", context)
print("Answer:", answer)
print("Generated Question:", question)
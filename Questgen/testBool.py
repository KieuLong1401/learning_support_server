from main import BoolQGen, AnswerPredictor

# if __name__ == "__main__":
#     bool_gen = BoolQGen()
    
#     payload = {
#         "input_text": "The Earth revolves around the Sun. Water boils at 100 degrees Celsius.",
#         "max_questions": 2
#     }
    
#     result = bool_gen.predict_boolq(payload)
#     print("Generated Questions:", result)
    
#     print("Input text:", payload["input_text"])
#     print("Generated Boolean Questions:")
#     for q in result["Boolean Questions"]:
#         print("-", q)
        
payload = {
    "input_text": "The Earth revolves around the Sun. Water boils at 100 degrees Celsius. sun and earth are not the same.",
    "input_question": ['Is the sun and the earth the same thing?']
}

answer_predictor = AnswerPredictor()

answer = answer_predictor.predict_answer(payload)[0]
print("Answer:", answer)
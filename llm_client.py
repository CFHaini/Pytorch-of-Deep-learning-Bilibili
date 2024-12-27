import openai
import sys

api_key = "kysxjsjzl"
openai.api_base = "http://127.0.0.1:6006/v1"
#  openai==20.8

def chat_with_gpt3_5(messages):
    response = openai.ChatCompletion.create(
        model="考研数学计算机之路",
        messages=messages,
        api_key=api_key,
        stream=True  # 启用流式输出
    )

    full_response = ""
    for chunk in response:
        if 'choices' in chunk and len(chunk['choices']) > 0:
            content = chunk['choices'][0].get('delta', {}).get('content', '')
            if content:
                print(content, end='', flush=True)
                full_response += content
    print() 
    return full_response

conversation = [
    {"role": "system", "content": "你是一个聪明的AI"}
]

while True:

    user_input = input("You: ")

    if user_input.lower() == '退出':
        print("Assistant: 再见！")
        break

    conversation.append({"role": "user", "content": user_input})

    print("Assistant: ", end='', flush=True)
    assistant_message = chat_with_gpt3_5(conversation)

    conversation.append({"role": "assistant", "content": assistant_message})


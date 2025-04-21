from ollama import Client
client = Client(
    host='http://192.168.137.188:11434',
)

# response = client.chat(model='gemma3:1b', messages=[
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue?',
#     },
# ])
#
# print(response)

# response = client.generate(model='gemma3:12b',
#                            # format='json', # Pi too slow for this?
#                            options={'seed': 42, 'temperature': .99},
#                            prompt = 'Who are the two main characters in "MacBeth"?')
#
# print(response.response)
# print((response.total_duration/1000)/60.0)
# response = client.generate(model='gemma3:12b',
#                            # format='json', # Pi too slow for this?
#                            options={'seed': 41, 'temperature': .4},
#                            prompt = 'Who are the two main characters in "MacBeth"?')
#
# print(response.response)
# print((response.total_duration/1000)/60.0)
# response = client.generate(model='gemma3:12b',
#                            # format='json', # Pi too slow for this?
#                            options={'seed': 40, 'temperature': .01},
#                            prompt = 'Who are the two main characters in "MacBeth"?')
#
# print(response.response)
# print((response.total_duration/1000)/60.0)
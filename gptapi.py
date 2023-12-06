from pathlib import Path
from gpt4all import GPT4All

def main(output_limit=1):
    model = GPT4All("mistral-7b-openorca.Q4_0.gguf")
    textA = input("What text do you want to test for?")
    wordcount = len(textA.split())
    i = 0
    textB = []
    with model.chat_session():
        type = model.generate(prompt=f'In 2 words state what type of text is this (Narrative Persuasive texts, Analytical texts, Argumentative, Article , Expository , Report, Discussion, Literature Review, Biography, Essay, Factual, recount , Instructive texts , Journalism , Letter ,Poetry ,Procedure, Short story) : {textA}', temp=0)
        person = model.generate(prompt=" In 1 word write whether it is a first-person, second-person, third-person text.")
    with model.chat_session():
        summary = model.generate(prompt=f"Write an extended summary of this text by finishing this  'about ... ,: {textA}", temp=0)
    while i < output_limit:
        with model.chat_session():
            textB1 = model.generate(prompt=f"Can you please write a complete {type} in {person} {summary} that is at least {wordcount} words long. Make sure to complete the {type}.", temp=1.0, max_tokens=300)
            wordcountGPT = len(textB1.split())
            if wordcountGPT > wordcount + 20 or wordcountGPT < wordcount - 20:
                textB1 = model.generate(prompt=f"Please rewrite it so it is at least {wordcount} words long and make sure to finish the whole {type}!", repeat_penalty=1.25, temp=1.0, max_tokens=300)
            i += 1
            textB.append(textB1.split())

    # print(textB1)
    # print(f"Type : {type}")
    # print(f"Summary : {summary}")
    output_filename = Path('llm_output.txt')
    open(output_filename, mode='w').write('\n\n===============\n\n'.join(textB))
    print(f'Successfully output to file {output_filename}')

if __name__ == '__main__':
    main(output_limit=5)

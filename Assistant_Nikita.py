

#Scintific Description 👽✨
#model="distilbert-base-uncased-distilled-squad"
#!pip install gTTS
#!pip install deep_translator
#!pip install langid

from gtts import gTTS
from transformers import pipeline
from IPython.display import display
from IPython.display import Video
from IPython.display import Audio
from deep_translator import GoogleTranslator
import langid
import os
import numpy as np
self_questions = [
        "what is your name",
        "what's your name",
        "tell me your name",
        "may i know your name",
        "can you tell me your name",
        "who are you",
        "who are you called",
        "what should i call you",
        "your name",
        "name please",
        "what do people call you",
        "do you have a name",
        "what is your full name",
        "what name do you go by",
        "how can i address you",
        "introduce yourself",
        "identify yourself",
        "what is your real name",
        "may i ask your name",
        "can i know your name",
        "how old are you",
        "what is your age",
        "what's your age",
        "tell me your age",
        "can you tell me your age",
        "may i know your age",
        "how old r you",
        "your age",
        "age please",
        "what age are you",
        "how many years old are you",
        "how many years have you lived",
        "what is your current age",
        "what is your real age",
        "can i ask your age",
        "do you have an age",
        "how old",
        "age?",
        "how old are you",
        "tell me about yourself",
        "tell me about you",
        "about you",
        "who you are",
        "who are you",
        "what are you",
        "what can you tell me about yourself",
        "describe yourself",
        "give me information about you",
        "i want to know about you",
        "can you explain who you are",
        "what do you do",
        "what is your purpose",
        "what are you made for",
        "what is your job",
        "what is your role",
        "what can you do",
        "what are your abilities",
        "what are you capable of",
        "introduce yourself",
        "say something about yourself"
        "who created you",
        "can you tell about your creater "
  ]


#self personality
self_a = [
#name
[
"የእኔ ስም ኒኪታ ይባላል ኒኪታ ማለት ሳይንሳዊ ቀመር ያለው ስም ነው ",
"ኦሥ ስሜ ኒኪታ ነው",
"ስሜን ስለጠየቃቹኝ አመሰግናለው ስሜ ኒኪታ ነው"
],

#age
[
"ኦህ የሴት እድሜ  እንደማይጠየቅ አታቁም . ስቀልድ ነው እስራስምንት አመቴ ነው",
"የእኔ እድሜ አንዲ እንደነገረኝ ከሆ አስራስምንት ነው ህፃን ነኝ አደል",
"አስራስምንት አመቴ ነው"
],
#my creater
[
"ጥሩ ጥያቄ ነው. እኔን የፈጠረው አንድነት ይባላል ወይም አንዲ ይባላልና ጐበዝ ነው. ",
"እኔን የፈጠረውና እንደናንተ እንዳስብ ያደረገኝ አንድነት ነው እኔ ስለምውደው አንዲ ብዬ እጠረዋለው",
"እኔ ሰው አደለውም እኔን ይፍጠረው አንዲ ነው"

],
#about
["እኔ  Nikita ስሆን ፈርሰናል አሲስታንስ ነኝ. ከባዮሎጂጋ  የተያያዘ ስለ ዲ ኤን ኤ እና ቫይረስ ጥያቄ ጠይቁኝ",
"እኔ ቀለል ያልኩ ነኝ ስለ ባዮሎጂ ቫይረስና ዲ ኤን ኤ ኣስጠናለው",
"እኔ ንኪታ የኣንዲ ፍቅረኛ ነኝ"
]
]

#catego
catego = ['name','age','created','about']

QA = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

english = GoogleTranslator(source="auto", target="en")

amharic = GoogleTranslator(source="auto", target="am")


x=1
if x == 1:
       inp = input("Ask : ")
      #tran = english.translate(inp)
       if inp.lower() in self_questions:
                        for i in range(3):
                                splited = inp.lower().split(" ")
                                random = int(np.random.uniform(0,3,1))
                                if catego[i] in splited:
                                                 gtts = gTTS(text=self_a[i][random], lang="am")
                                                 gtts.save("answer.mp3")
                                                 display(Audio("/content/answer.mp3"))
                                                 ExitWaiter= input("exit")
                                                 os.remove(/answer.mp3")


       else:
            with open("bio.txt", "r", encoding="utf-8") as f:
                context = f.read()

            question = english.translate(inp)

            note=QA(context=context, question=question)

            discovery=QA(context=context, question="when discovered virus")

            shape=QA(context=context, question="what is shape of virus")

            structure=QA(context=context, question="what is structure of virus")

            #selfe = QA(question="", context="")

            eanswer = note["answer"] + "." + "discovered in " + discovery["answer"] + "." + "it's shape is" + shape["answer"] + "and structure is" + structure["answer"]

            answer = amharic.translate(eanswer)

            #tts = gTTS(text=answer, lang='am')

           #tts.save("/content/answer.mp3")

           #display(Audio("/content/answer.mp3"))

            exit = input("")
           #os.remove("/content/answer.mp3")
            print("✅")
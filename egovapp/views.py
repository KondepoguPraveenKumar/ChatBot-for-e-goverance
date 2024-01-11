from django.shortcuts import render,redirect
from django.http import HttpResponse
# Create your views here.
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from .basic import Basic


bot_rea = Basic.quaAns()


def index(req):
    return render(req,'index.html')
 

def getres(req):
    u_msg = req.GET.get('msg')
    # You can now use the chatbot to generate responses
    user_message = u_msg
    response = bot_rea.get_response(user_message)
    print(response)
    return HttpResponse(response)
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pandas as pd
from .bert import BertAlgo




class Basic:
    def quaAns():
        bot = ChatBot('MyBot')
        # Create a ListTrainer instance and train the chatbot
        list_trainer = ListTrainer(bot)
        df = BertAlgo.bert()
        conversations = [
            'hi',
            'Hi how can i help you',
            'Hello',
            'Hello How are you?',
            'fine',
            'which bank information you want RBI or NABARD?',
            'RBI',
            'Which scheme information you want ?',
            'bima yojana',
            'In order to enrol in PMSBY offline, one can visit the bank branch where one has a savings account or the candidate can visit the official site:https://jansuraksha.gov.in/Forms-PMSBY.aspx to download the form.After downloading the application form ,candidate can fill in all the details and submit  them to the bank alongwith the  required documents.Once it is successfully submitted,subscriber will get an Acknowledgement Slip Cum Certificate of Insurance.',
        ]
        
        for i in df['title']:
            conversations.append(str(i))
            for j in df['answer']:
                conversations.append(str(j))

        # Train the chatbot with the list of conversations
        list_trainer.train(conversations)
        
        # import csv

        # conversation_pairs = []
        # reader = df
        # for row in reader:
        #     conversation_pairs.append((row['title'], row['answer']))
        # list_trainer.train(conversation_pairs)
        return bot
    
    
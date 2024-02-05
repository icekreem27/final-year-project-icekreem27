from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

bot = ChatBot('testBot')
trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.english")

print("Hello, type something to begin...")

while True:
    try:
        user_input = input("User> ")
        if user_input.lower() == 'bye':
            print("ChatBot> bye!")
            break
        response = bot.get_response(user_input)
        print(f"ChatBot> {response}")
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("ChatBot> \nbye!")
        break

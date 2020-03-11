#!/usr/bin/env python3

config = GPT2Config(n_ctx=1024, n_embd=1024, n_layer=24, n_head=16)
tokenizer = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config).from_pretrained("medium_ft.pkl")

while True:
    user_input = input(">> User:")

    if user == "quit":
        print("Good bye!")
        break

    user_input_ids = tokenizer.encode(user_input, return_tensors='pt')
    input_ids = torch.cat([input_ids, user_input_ids], dim=-1)
    bot_input_ids = model.generate(input_ids, max_length=300)
    print("DialoGPT: {}".format(tokenizer.decode(bot_input_ids[0])))

# Example for GPT2LMHeadModel

# CURRENLTY:
outputs = model(input_ids)

# To get certain outputs
logits = outputs[0] # ...but could also be logits = outputs[1] if loss is defined
past = outputs[1] 

# PROPOSITION: use namedtuple
from collection import namedtuple

# Define namedtuple
Outputs = namedtuple('Outputs', ['logits', 'past', 'attentions'])

# Return in forward() function of GPTLMHeadModel an namedtuple
# create namedtuple
# logits = [1, 2] - two logits
# past = ([0,0], [1,5]) - two layers
# attentions = ([7, 8], [4, 7])
outputs = Outputs(logits, past, attentions)

# THEN...

outputs = model(input_ids)

logits = outputs[0] = outputs.logits # backward compatible and can also be accesed via outputs.logits
past = outputs[1] = outputs.past  # ...
attentions = outputs[2] = outputs.attentions  # ...

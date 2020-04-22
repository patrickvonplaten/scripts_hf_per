#!/usr/bin/env python3
import nlp
ds = nlp.load("crime_and_punish", split='train[:1%]')

import ipdb
ipdb.set_trace()

new_ds = ds.map(lambda batch: {'paragraph': ['\n'.join(batch['paragraph'])]}, batched=True, batch_size=20, load_from_cache_file=False)


import ipdb
ipdb.set_trace()

pass

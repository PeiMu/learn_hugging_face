from transformers import AutoConfig
from transformers import AutoModel

my_config = AutoConfig.from_pretrained("distilbert-base-uncased", n_heads=12)
my_model = AutoModel.from_config(my_config)
print(my_model)


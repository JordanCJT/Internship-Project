from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
print("✅ Tokenizer downloaded successfully")

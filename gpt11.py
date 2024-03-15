
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ['NO_PROXY'] = 'huggingface.co'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2TokenizerFast

ticker = "NSEI"
start_date = "2023-01-01"
end_date = "2023-06-08"
data = yf.download(ticker, start=start_date, end=end_date)
prices = data["Close"].tolist()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


encoded_prices = tokenizer.encode(" ".join([str(price) for price in prices]), return_tensors="pt")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.resize_token_embeddings(len(tokenizer))
for _ in range(3):
    model.zero_grad()
    outputs = model(encoded_prices, labels=encoded_prices)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, prices, label="Historical Prices")
    plt.plot(data.index[-1] + pd.to_timedelta(np.arange(1, len(generated_prices) + 1), 'D'),
             [float(price) for price in generated_prices[len(prices):]], "g^", label="Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} - Historical and Predicted Stock Prices (GPT)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
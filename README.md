# Cryptocurrency predictions
This is a project that I completed during a machine learning bootcamp at UCSD.

All three of the models in models.py have been used to predict the prices of nine different popular cryptocurrencies, and the data has been pulled using the YahooFinanceAPI.

The first model is a simple LSTM, not quite the same as have two stacked cells, but something similar. It does predict, but instead of guessing at what prices will be next, it seems that it usually lags behind a day. This isn't the case for all of the coins, but it is the case for some.

The second model is a vanilla transformer model, the same that is introduced in "Attention is All You Need" [1]. It has been modified slightly, still using 4 encoder/decoder layers, but the model is not nearly as deep as when it was originally displayed. Since the data is just time series, and not very long, having a much larger transformer was leading to overfitting of the train set. Though the model itself is the same as in [1], this specific implementation is the one used in "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case" [4].

The third model is something that the authors of "Are Transformers Effective for Time Series Forecasting?" [2] have named DLinear. It consists of a linear neural network that uses series decomposition and a moving average to make its predictions. In the paper, the model is meant to be used for long time series forecasting for more predictable data, such as energy usage. Although it doesn't perform as well as the transformer on the crypto price data, it does still do quite well, and takes only a fraction of the time and space of a transformer model.

I had planned on implementing the model in [3], which used residuals from the ARIMA model as input for an LSTM, as well as used XGBoost as a decoder for the predictions of the LSTM. This seemed rather interesting, however after testing their code I realized that it does not function they way they explain it in the paper. I was able to implement it the way I believe they intended, however my implementation was incredibly slow, and only performed slightly better.


If you are interested in trying any of these models for yourself, you can run database_setup.py, and then first_training.py, and then adding_first_predictions.py. You can change those dates directly in the python file, and you can add or take away coins.

References: 

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/ARXIV.1706.03762

[2] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022). Are Transformers Effective for Time Series Forecasting? arXiv. https://doi.org/10.48550/ARXIV.2205.13504

[3] Shi, Z., Hu, Y., Mo, G., & Wu, J. (2022). Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction. arXiv. https://doi.org/10.48550/ARXIV.2204.02623

[4] Wu, N., Green, B., Ben, X., O'banion, S. (2020). Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case. arXiv. https://doi.org/10.48550/arXiv.2001.08317
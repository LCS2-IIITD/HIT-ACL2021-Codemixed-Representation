# README for HIT

### Installation for experiments

	$ pip install -r requirements.txt

### Commands to run
	
	$ cd experiments

	Sentiment analysis

	$ python experiments_hindi_sentiment.py --train_data ../data/hindi_sentiment/IIITH_Codemixed.txt --model_save_path ../models/model_hindi_sentiment/

	POS

	$ python experiments_hindi_POS.py --train_data '../data/POS Hindi English Code Mixed Tweets/POS Hindi English Code Mixed Tweets.tsv' \
									  --model_save_path ../models/model_hindi_pos/

    NER

    $ python experiments_hindi_NER.py --train_data '../data/NER/NER Hindi English Code Mixed Tweets.tsv' \
									  --model_save_path ../models/model_hindi_NER/

	MT

	$ python nmt.py --data_path '../data/IITPatna-CodeMixedMT' --model_save_path ../models/model_hindi_NMT/

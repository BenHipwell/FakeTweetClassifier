# FakeTweetClassifier
Using machine learning &amp; NLP to identify fake tweets using the mediaeval-2015 dataset

1. Ensure all libraries that have been imported are installed - all of them are available via pip.
2. The test and train data sets need to be within the same folder/file location for it to be read, with their original names.
3. Uncomment the first line of code, on line 21, for the first run to download the sentiment analysis model.  
4. The current state of the script is that of the run with translation disabled (function call commented out on line 77), the classifier is
the most successful one from my evaluation (Multinomial NB) and Select K Best is enabled, with K set to 4000.
5. To remove the Select K Best feature selection, comment out lines 113 and 125 at the least.

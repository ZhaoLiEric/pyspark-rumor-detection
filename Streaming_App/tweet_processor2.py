import re
from functools import partial

import findspark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IDF, HashingTF, StopWordsRemover, Tokenizer, VectorAssembler, StandardScaler
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, window, size, to_timestamp, unix_timestamp, col, udf, expr, count, \
    regexp_replace
from pyspark.sql.types import Row, IntegerType, FloatType, StringType


host = "localhost"
port = 9999

findspark.init()

spark = SparkSession.builder \
    .getOrCreate()
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
sc = spark.sparkContext

temp = spark.read.json("sample_tweet.json")
schema = temp.schema


socketDF = spark.readStream.format('socket') \
    .option('host', host) \
    .option('port', port) \
    .load()

tweets = socketDF. \
    selectExpr("CAST(value AS STRING)"). \
    select(from_json("value", schema).alias('tmp')).select("tmp.*")

tweets = tweets.withColumn('timestamp', to_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy"))

words_dict = dict(
    has_belief_words=set("assume believe apparent per-haps suspect think thought consider".split()),
    has_report_words=set("evidence source official footage capture assert told claim according".split()),
    has_doubt_words=set("wonder allege unsure guess speculate doubt".split()),
    has_knowledge=set("confirm definitely admit".split()),
    has_denial_words=set("refuse reject rebuff dismiss contradict oppose".split()),
    has_curse_words=set("lol rofl lmfao yeah stfu aha wtf shit".split()),
    # negation_words = set("no not no one nothing never don’t can’t hardly".split(),
    has_question_words=set("when which what who how whom why whose".split()),
    has_other_words=set("irresponsible careless liar false witness untrue neglect integrity murder fake".split())
)
### DATA CLEANING  ##
def clean(df):

    def replace_url(text):
        return re.sub(r'https?:\/\/.*[\r\n]*', 'url_url_url', text, flags=re.MULTILINE)

    replace_url_udf = udf(replace_url, StringType())

    df = df.withColumn('cleaned_text', replace_url_udf(col('text')))

    ### REMOVE @
    df = df.withColumn('cleaned_text', regexp_replace(col('cleaned_text'), r'(@([A-Za-z0-9]+))', ''))

    return df

## FEATURE EXTRACTION ##

def extract_features(df):
    hasqmark = udf(lambda x: int('?' in x), IntegerType())
    df = df.withColumn('hasqmark', hasqmark(col('cleaned_text')))

    hasmark = udf(lambda x: int('!' in x), IntegerType())
    df = df.withColumn('hasmark', hasmark(col('cleaned_text')))

    hasperiod = udf(lambda x: int('.' in x), IntegerType())
    df = df.withColumn('hasperiod', hasperiod(col('cleaned_text')))

    df = df.withColumn('hashtags_count', expr('size(entities.hashtags)'))

    df = df.withColumn('mentions_count', expr('size(entities.user_mentions)'))

    df = df.withColumn('hasurls', expr('cast(size(entities.urls) >= 1 AS int)'))

    # df = df.withColumn('hasmedia', expr('cast(size(entities.media) >= 1 AS int)'))

    df = df.withColumn('friends_count', expr('user.friends_count'))

    df = df.withColumn('followers_count', expr('user.followers_count'))

    ratiocapital = udf(lambda x: sum(map(str.isupper, x))/(len(x)+1), FloatType())
    df = df.withColumn('ratiocapital', ratiocapital(col('cleaned_text')))

    charlen = udf(lambda x: len(x), IntegerType())
    df = df.withColumn('charlen', charlen(col('cleaned_text')))

    df = df.withColumn('issource', expr('CAST((in_reply_to_status_id IS NULL) AS INT)'))

    ## TOKENIZATION ##

    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    temp_df = tokenizer.transform(df)


    #--eric--

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = remover.transform(temp_df)


    wordlen = udf(lambda words: len(words), IntegerType())
    df = df.withColumn('wordlen', wordlen(col('words')))

    def contains(y, x):
      return int(bool(len(y.intersection(set(x)))))


    for name, ys in words_dict.items():
        df = df.withColumn(name, udf(partial(contains, ys), IntegerType())(col('words')))



    #TODO negation words etc.
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
              'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
              'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
              'couldn', 'doesn']
    def negacount(words):
      c = 0
      for negationword in negationwords:
        if negationword in words:
          c += 1
      return c
    negationcount = udf(negacount, IntegerType())
    df = df.withColumn('hasnegation', negationcount(col('words')))

    @udf('float')
    def count_upper(x):
        a = x.split()
        return sum(map(str.isupper, a))/(len(a) + 1)

    df = df.withColumn('allcapsratio', count_upper(col('cleaned_text')))


    return df




tweets = extract_features(clean(tweets))

model = PipelineModel.load('subtaskA_model')
y_pred = model.transform(tweets).select(['timestamp', 'prediction'])
out = y_pred.groupby(window(tweets.timestamp, '10 seconds', '10 seconds'), col('prediction')).count()
out = out.orderBy('window', 'prediction')

out \
    .writeStream.outputMode('complete') \
    .format('console') \
    .option('truncate', 'false') \
    .start() \
    .awaitTermination()

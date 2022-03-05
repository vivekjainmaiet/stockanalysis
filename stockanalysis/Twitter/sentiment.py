
neg_words = [
    'short', 'sell', 'sold', 'scary', 'ass', 'suck', 'crash', 'dump', 'shit',
    'loose', 'lost', 'not', 'bad', 'low', 'negative', 'fear', 'afraid',
    'worse', 'down', 'worry', 'fail', 'failure', 'worst', 'crush', 'fuck',
    'crushed', 'bear', 'war', 'weak', 'waste', 'sad', 'drop', 'sadly',
    'stupid', 'against', 'storm', 'warn', 'broke', 'red', '😠', '😡', '🤬', '🤢',
    '🤮', '💩', '😢', '😭', '😤', '😰', '😥', '😓', '😱', '😨', '😣', '📉'
]


pos_words = [
    'long', 'buy', 'bought', 'hold', 'hodl', 'top', 'increase', 'boom', 'wow',
    'moon', 'good', 'happy', 'win', 'won', 'gain', 'grow', 'strong', 'beauty',
    'great', 'positive', 'rise', 'bump', 'amazing', 'rebound', 'awesome',
    'opportunity', 'green', 'discount', 'opportunities', '😀', '😃', '😄', '😁',
    '😆', '😊', '😇', '🙂', '🤑', '💗', '😎', '😍', '🥰', '😎', '🤩', '😋', '😛', '😝', '😜',
    ' 🚀', '📈'
]

def custom_sentiment_base(text):
    pos_count=0
    neg_count=0

    for word in neg_words:
        if word in (text):
            neg_count+=1

    for word in pos_words:
        if word in (text):
            pos_count+=1

    if pos_count>neg_count:
        return 'pos'

    if neg_count>pos_count:
        return 'neg'

    else:
        return 'neu' #equal

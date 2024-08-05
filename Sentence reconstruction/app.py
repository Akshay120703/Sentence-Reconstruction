from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
import language_tool_python

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer and LanguageTool
sia = SentimentIntensityAnalyzer()
tool = language_tool_python.LanguageTool('en-US')

app = Flask(__name__)

# Function to get sophisticated synonym
def get_sophisticated_synonym(word, pos):
    synsets = wn.synsets(word, pos=pos)
    if synsets:
        for synset in synsets:
            sophisticated_word = synset.lemmas()[0].name().replace('_', ' ')
            if sophisticated_word.lower() != word.lower():
                return sophisticated_word
    return word

# POS mapping from NLTK to WordNet
pos_map = {
    'NN': wn.NOUN,
    'VB': wn.VERB,
    'JJ': wn.ADJ,
    'RB': wn.ADV
}

# Function to enhance sentence with positive tone
def enhance_sentence(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    enhanced_sentence = []
    for word, tag in tagged_tokens:
        wn_tag = pos_map.get(tag[:2])
        if wn_tag:
            enhanced_word = get_sophisticated_synonym(word, wn_tag)
            enhanced_sentence.append(enhanced_word)
        else:
            enhanced_sentence.append(word)
    return ' '.join(enhanced_sentence)

# Function to adjust the tone of the sentence to be more positive and kind-hearted
def make_kind_hearted(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    enhanced_sentence = []
    for word, tag in tagged_tokens:
        wn_tag = pos_map.get(tag[:2])
        if wn_tag:
            enhanced_word = get_sophisticated_synonym(word, wn_tag)
            if sia.polarity_scores(enhanced_word)['compound'] >= 0:
                enhanced_sentence.append(enhanced_word)
            else:
                enhanced_sentence.append(word)
        else:
            enhanced_sentence.append(word)
    return ' '.join(enhanced_sentence)

# Function to check and correct grammar
def correct_grammar(sentence):
    matches = tool.check(sentence)
    corrected_sentence = language_tool_python.utils.correct(sentence, matches)
    return corrected_sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    sentence = request.form.get('sentence', '')
    enhanced = enhance_sentence(sentence)
    kind_hearted = make_kind_hearted(enhanced)
    grammatically_correct = correct_grammar(kind_hearted)
    return jsonify({
        'enhanced': enhanced,
        'kind_hearted': kind_hearted,
        'grammatically_correct': grammatically_correct
    })

if __name__ == '__main__':
    app.run(debug=True)

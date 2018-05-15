from normalization import parse_document
from utils import build_feature_matrix, low_rank_svd
from gensim.summarization import summarize, keywords
import numpy as np


document1 = """
The patient is an 86-year-old female admitted for evaluation of abdominal pain and bloody stools. The patient has colitis and also diverticulitis, undergoing treatment.
During the hospitalization, the patient complains of shortness of breath, which is worsening. The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large pleural effusion.
This consultation is for further evaluation in this regard. As per the patient, she is an 86-year-old female, has limited activity level.
She has been having shortness of breath for many years. She also was told that she has a heart murmur, which was not followed through on a regular basis.
"""

document2 = """
Elephants are large mammals of the family Elephantidae 
and the order Proboscidea. Two species are traditionally recognised, 
the African elephant and the Asian elephant. Elephants are scattered 
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male 
African elephants are the largest extant terrestrial animals. All 
elephants have a long trunk used for many purposes, 
particularly breathing, lifting water and grasping objects. Their 
incisors grow into tusks, which can serve as weapons and as tools 
for moving objects and digging. Elephants' large ear flaps help 
to control their body temperature. Their pillar-like legs can 
carry their great weight. African elephants have larger ears 
and concave backs while Asian elephants have smaller ears 
and convex or level backs.  
"""

def text_summarization_gensim(text, summary_ratio=0.5):    
    summary = summarize(text, split=True, ratio=summary_ratio)
    for sentence in summary:
        print (sentence)

		
# Using Gensim Summarization Method
docs = parse_document(document1)
text = ' '.join(docs)
text_summarization_gensim(text, summary_ratio=0.3)
    
	
sentences = parse_document(document1)
norm_sentences = normalize_corpus(sentences,lemmatize=False) 

total_sentences = len(norm_sentences)
print ('Total Sentences in Document:', total_sentences)   

num_sentences = 3
num_topics = 1

vec, dt_matrix = build_feature_matrix(sentences, feature_type='frequency')

td_matrix = dt_matrix.transpose()
td_matrix = td_matrix.multiply(td_matrix > 0)

u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
                                         
sv_threshold = 0.5
min_sigma_value = max(s) * sv_threshold
s[s < min_sigma_value] = 0

salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
print (np.round(salience_scores, 2))

top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
top_sentence_indices.sort()
print (top_sentence_indices)

for index in top_sentence_indices:
    print (sentences[index])
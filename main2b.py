from oblig2b import compute_bleu # no idea why, but only import code variant that works for me
from oblig2b import RetrievalChatbot

print("Uten frasetabell: ", compute_bleu('lotr.en', 'lotr.full.out_no_dic.en'))
print("Med frasetabell: ", compute_bleu('lotr.en', 'lotr.full.out_with_dic.en'))

chatbot = RetrievalChatbot('lotr.en')
print("Response:", chatbot.get_response('Are you Bilbo Baggins ?'))
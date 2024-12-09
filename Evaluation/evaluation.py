from nltk.translate.bleu_score import sentence_bleu

reference = "The cat is on the mat."
candidate = "The cat is sitting on the mat."

score = sentence_bleu([reference.split()], candidate.split())
print("BLEU Score:", score)


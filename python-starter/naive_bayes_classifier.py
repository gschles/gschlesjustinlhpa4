from __future__ import print_function

import sys, math, heapq

from time import clock

from collections import Counter, defaultdict
from message_iterators import MessageIterator





m = 2.0
k = 10

class IteratorInfo:
        def __init__(self, numgroups, num_msgs = None, tot_msgs = None, messages = None):
                self.numgroups = numgroups
                self.num_msgs = [0] * numgroups if num_msgs is None else num_msgs
                self.tot_msgs = 0 if tot_msgs is None else tot_msgs
                self.messages = [] if messages is None else messages

"""
train_binomial() counts the number of documents each word 
occurs for each class
"""
def train_binomial(ii, V, class_counters):
  N = ii.tot_msgs
  prior = [float(ii.num_msgs[c])/N for c in range(ii.numgroups)] # class priors
  condprob = [{} for c in range(ii.numgroups)]
  noprob = [math.log(prior[c]) for c in range(ii.numgroups)]
  for term in V:
    ovf = float(sum([class_counters[c][term] for c in range(ii.numgroups)]))/ii.tot_msgs
    for c in range(ii.numgroups):
      condprob[c][term] = float(class_counters[c][term]+1)/float(ii.num_msgs[c]+2)
      #condprob[c][term] = float(class_counters[c][term]+m*of)/float(mi.num_msgs[c]+m)
      noprob[c] += math.log(1-condprob[c][term])
  return prior, condprob, noprob
"""
given a document class and model, computes the probability
of the first 20 docs for each class
"""
def apply_binomial(C, V, prior, condprob, noprob, doc):
  scores = [noprob[c] for c in range(C)]
  for c in range(C):
    s = set([])
    for term in doc.body:
      if term in V:
        s.add(term)
        scores[c] += math.log(condprob[c][term]) - math.log(1-condprob[c][term])
  n = 0
  if scores.index(max(scores)) == doc.newsgroupnum:
    n = 1
  #print(str(doc.newsgroupnum)+' '+str(scores.index(max(scores))), file=sys.stderr)
  return scores, n

def parse_first_20(ii):
  docs = []
  count = 0
  group = 0
  for mf in ii.messages:
    if group >= ii.numgroups:
      break
    if mf.newsgroupnum == group:
      docs.append(mf)
      count += 1
    if count >= 20:
      group += 1
      count = 0
  return docs

def binomial(ii):
  (V, class_counters) = extract_vocab(ii)
  (prior, condprob, noprob) = train_binomial(ii, V, class_counters)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(ii)
  print('got test docs', file=sys.stderr)
  cor = 0
  for i, doc in enumerate(test_docs):
    (probs, n) = apply_binomial(ii.numgroups, V, prior, condprob, noprob, doc)
    cor += n
    output_probability(probs)
  print(float(cor)/400, file=sys.stderr)

def extract_vocab(ii):
  V = set([])
  class_counters = [Counter() for c in range(ii.numgroups)]
  for mf in ii.messages:
    for word in mf.body:
      class_counters[mf.newsgroupnum][word] += 1
      V.add(word)
  return V, class_counters

def calc_chi2(mi, class_counts, c, term):
  N11 = class_counts[c][term]
  N10 = sum([class_counts[k][term] for k in range(mi.numgroups) if k != c])
  N01 = mi.num_msgs[c] - N11
  N00 = mi.tot_msgs - mi.num_msgs[c] - N10
  num = float((N11+N10+N01+N00)*math.pow(N11*N00-N10*N01,2))
  denom = float((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))
  return num/denom

def extract_chi_square(mi, V, class_counts, c):
  chi2 = []
  for term in V:
    heapq.heappush(chi2, (calc_chi2(mi, class_counts, c, term), term))
  return [h[1] for h in heapq.nlargest(300, chi2)]

def extract_features(mi, V, class_counts):
  features = set([])
  for c in range(mi.numgroups):
    features.update(extract_chi_square(mi, V, class_counts, c))
  return features

def binomial_chi2(mi):
  (V, class_counters) = extract_vocab(mi)
  features = extract_features(mi, V, class_counters)
  (prior, condprob, noprob) = train_binomial(mi, features, class_counters)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(mi)
  print('got test docs', file=sys.stderr)
  cor = 0
  for i, doc in enumerate(test_docs):
    (probs, n) = apply_binomial(mi.numgroups, features, prior, condprob, noprob, doc)
    cor += n
    output_probability(probs)
  print(float(cor)/400, file=sys.stderr)
"""
trains the multinomial model
"""
def train_multinomial(ii):
  V = set([])
  N = ii.tot_msgs
  class_counters = [Counter() for i in range(ii.numgroups)]
  for mf in ii.messages:
    for word, count in mf.body.items():
      class_counters[mf.newsgroupnum][word] += count
      V.add(word)
  prior = [float(ii.num_msgs[c])/ii.tot_msgs for c in range(ii.numgroups)] # class priors
  condprob = [{} for i in range(ii.numgroups)]
  for c in range(ii.numgroups):
    denom = sum([class_counters[c][t]+1 for t in class_counters[c]])
    for term in V:
      condprob[c][term] = float(class_counters[c][term]+1)/denom
  return V, prior, condprob

"""
given a document class and model, computes the probability
of the first 20 docs for each class
"""
def apply_multinomial(C, V, prior, condprob, doc):
  scores = [0 for c in range(C)]
  for c in range(C):
    scores[c] = math.log(prior[c]) #class prior prob
    for term, count in doc.body.items():
        if term in condprob[c]:
         scores[c] += count*math.log(condprob[c][term])
  n = 0
  if scores.index(max(scores)) == doc.newsgroupnum:
    n = 1
  #print(str(doc.newsgroupnum)+' '+str(scores.index(max(scores))), file=sys.stderr)
  return scores, n

def multinomial(ii):
  (V, prior, condprob) = train_multinomial(ii)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(ii)
  print('got test docs', file=sys.stderr)
  #for doc in test_docs:
   # output_probability(apply_multinomial(mi.numgroups, V, prior, condprob, doc))
  cor = 0
  for doc in test_docs:
    (probs, n) = apply_multinomial(ii.numgroups, V, prior, condprob, doc)
    cor += n
    output_probability(probs)
  print(float(cor)/400, file=sys.stderr)


def apply_twcnb(C, prior, comp_condprob, class_weights, doc):
  scores = [0 for c in range(C)]
  for c in range(C):
    for term, count in doc.body.items():
        if term in comp_condprob[c]:
                weight = math.log(comp_condprob[c][term])
                #Modifying weight frequencies to implement TWCNB
                frequency = math.log(1 +count)
                scores[c] -= frequency * weight
    scores[c] /= class_weights[c]
    #scores[c] += math.log(prior[c]) #class prior prob
  n = 0
  if scores.index(max(scores)) == doc.newsgroupnum:
   n = 1
  return scores, n


def train_twcnb(ii):
   V = set([])
   N = ii.tot_msgs
   class_counters = [Counter() for i in range(ii.numgroups)]
   num_words = 0;
   word_counts = Counter()
   word_messages_count = Counter()
   class_weights = Counter()
   for mf in ii.messages:
     for word, count in mf.body.items():
       class_counters[mf.newsgroupnum][word] += count
       word_messages_count[word] += 1
       num_words +=count
       word_counts[word] += count
       V.add(word)
   prior = [float(ii.num_msgs[c])/ii.tot_msgs for c in range(ii.numgroups)] # class priors
   comp_condprob = [{} for i in range(ii.numgroups)]
   for c in range(ii.numgroups):
     denom = num_words - sum([class_counters[c][t]+1 for t in class_counters[c]])
     #implement CNB using word_counts, class weights aids in WCNB
     for term in V:
       comp_condprob[c][term] = float((word_counts[term] - class_counters[c][term])+1)/denom
       class_weights[c] += math.fabs(math.log(comp_condprob[c][term]))
   return prior, comp_condprob, class_weights

def twcnb(ii):
  (prior, comp_condprob,class_weights) = train_twcnb(ii)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(ii)
  print('got test docs', file=sys.stderr)
  cor = 0
  for doc in test_docs:
    (probs, n) = apply_twcnb(ii.numgroups, prior, comp_condprob, class_weights, doc)
    cor += n
    #output_probability(probs)
  print(float(cor)/400, file=sys.stderr)



def split_main_iterator(ii):
    kfold_message_iterators = [IteratorInfo(ii.numgroups,None,None,None) for i in range(k)]
    for i, mf in enumerate(ii.messages):
        index = i % k
        kfold_message_iterators[index].messages.append(mf)
        kfold_message_iterators[index].tot_msgs += 1
        kfold_message_iterators[index].num_msgs[mf.newsgroupnum] +=1
    return kfold_message_iterators

def merge_kfold_message_iterators(kfold_message_iterators,j,numgroups):
    train_message_iterator = IteratorInfo(numgroups,None,None,None)
    for i, iterator in enumerate(kfold_message_iterators):
        if i != j:
                train_message_iterator.messages += iterator.messages
                train_message_iterator.tot_msgs += iterator.tot_msgs
                for c in range(numgroups):
                        train_message_iterator.num_msgs[c] += iterator.num_msgs[c]
    return train_message_iterator, kfold_message_iterators[j]


def kfold_binomial(train_message_iterator, test_message_iterator):
 (V, class_counters) = extract_vocab(train_message_iterator)
 (prior, condprob, noprob) = train_binomial(train_message_iterator, V, class_counters)
 cor = 0.0
 for i, message in enumerate(test_message_iterator.messages):
      (probs, n) = apply_binomial(test_message_iterator.numgroups, V, prior, condprob, noprob, message)
      cor += n
 print ("b",cor, test_message_iterator.tot_msgs)
 return cor/test_message_iterator.tot_msgs

def kfold_multinomial(train_message_iterator, test_message_iterator):
  (V, prior, condprob) = train_multinomial(train_message_iterator)
  cor = 0.0
  for i,message in enumerate(test_message_iterator.messages):
     (probs, n) = apply_multinomial(test_message_iterator.numgroups, V, prior, condprob, message)
     cor += n
  print("m",cor,test_message_iterator.tot_msgs)
  return cor/test_message_iterator.tot_msgs

def kfold(ii):
  binom_tot_acc = 0
  multinom_tot_acc = 0
  kfold_message_iterators = split_main_iterator(ii)
  for j in range(k):
      (train_message_iterator,test_message_iterator) = merge_kfold_message_iterators(kfold_message_iterators,j,ii.numgroups)
      binom_tot_acc += kfold_binomial(train_message_iterator, test_message_iterator)
      multinom_tot_acc += kfold_multinomial(train_message_iterator, test_message_iterator)
  print("binomial accuracy", binom_tot_acc/k)
  print("multinomial accuracy", multinom_tot_acc/k)


def output_probability(probs):
  for i, prob in enumerate(probs):
    if i == 0:
      sys.stdout.write("{0:1.8g}".format(prob))
    else:
      sys.stdout.write("\t{0:1.8g}".format(prob))
  sys.stdout.write("\n")


MODES = {
    'binomial': binomial,
    'binomial-chi2': binomial_chi2,
    'multinomial': multinomial,
    'twcnb': twcnb,
    'kfold' : kfold
    # Add others here if you want
    }

def main():
  start = clock()
  if not len(sys.argv) == 3:
    print("Usage: python {0} <mode> <train>".format(__file__), file=sys.stderr)
    sys.exit(-1)
  mode = sys.argv[1]
  train = sys.argv[2]

  mi = MessageIterator(train)
  ii = IteratorInfo(mi.numgroups,mi.num_msgs,mi.tot_msgs,[mf for mf in mi])

  try:
    MODES[mode](ii)
    print(str(clock()-start),file=sys.stderr)
  except KeyError:
    print("Unknown mode: {0}".format(mode),file=sys.stderr)
    print("Accepted modes are: {0}".format(MODES.keys()), file=sys.stderr)
    sys.exit(-1)
     
if __name__ == '__main__':
  main()
#assumes body_hits exists for each query term 
#assumes body_hits exists for each query term 
#assumes body_hits exists for each query term 

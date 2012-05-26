
from __future__ import print_function

import sys, math, heapq

from time import clock

from collections import Counter
from message_iterators import MessageIterator

m = 2.0

"""
train_binomial() counts the number of documents each word 
occurs for each class
"""
def train_binomial(mi, V, class_counters):
  N = mi.tot_msgs
  prior = [float(mi.num_msgs[c])/N for c in range(mi.numgroups)] # class priors
  condprob = [{} for c in range(mi.numgroups)]
  noprob = [math.log(prior[c]) for c in range(mi.numgroups)]
  for term in V:
    ovf = float(sum([class_counters[c][term] for c in range(mi.numgroups)]))/mi.tot_msgs
    for c in range(mi.numgroups):
      condprob[c][term] = float(class_counters[c][term]+1)/float(mi.num_msgs[c]+2)
      #condprob[c][term] = float(class_counters[c][term]+m*ovf)/float(mi.num_msgs[c]+m)
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

def parse_first_20(mi):
  docs = []
  count = 0
  group = 0
  for mf in mi:
    if group >= mi.numgroups:
      break
    if mf.newsgroupnum == group:
      docs.append(mf)
      count += 1
    if count >= 20:
      group += 1
      count = 0
  return docs

def binomial(mi):
  (V, class_counters) = extract_vocab(mi)
  (prior, condprob, noprob) = train_binomial(mi, V, class_counters)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(mi)
  print('got test docs', file=sys.stderr)
  cor = 0
  for i, doc in enumerate(test_docs):
    (probs, n) = apply_binomial(mi.numgroups, V, prior, condprob, noprob, doc)
    cor += n
    output_probability(probs)
  print(float(cor)/400, file=sys.stderr)
  
def extract_vocab(mi):
  V = set([])
  class_counters = [Counter() for c in range(mi.numgroups)]
  for mf in mi:
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
def train_multinomial(mi):
  V = set([])
  N = mi.tot_msgs
  class_counters = [Counter() for i in range(mi.numgroups)]
  for mf in mi:
    for word, count in mf.body.items():
      class_counters[mf.newsgroupnum][word] += count
      V.add(word)
  prior = [float(mi.num_msgs[c])/mi.tot_msgs for c in range(mi.numgroups)] # class priors
  condprob = [{} for i in range(mi.numgroups)]
  for c in range(mi.numgroups):
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
      scores[c] += count*math.log(condprob[c][term])
  n = 0
  if scores.index(max(scores)) == doc.newsgroupnum:
    n = 1
  #print(str(doc.newsgroupnum)+' '+str(scores.index(max(scores))), file=sys.stderr)
  return scores, n
  
def multinomial(mi):
  (V, prior, condprob) = train_multinomial(mi)
  print('done training', file=sys.stderr)
  test_docs = parse_first_20(mi)
  print('got test docs', file=sys.stderr)
  #for doc in test_docs:
   # output_probability(apply_multinomial(mi.numgroups, V, prior, condprob, doc))
  cor = 0
  for doc in test_docs:
    (probs, n) = apply_multinomial(mi.numgroups, V, prior, condprob, doc)
    cor += n
    output_probability(probs)
  print(float(cor)/400, file=sys.stderr)

def twcnb(mi):
  pass

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
    'twcnb': twcnb
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

  try:
    MODES[mode](mi)
    print(str(clock()-start),file=sys.stderr)
  except KeyError:
    print("Unknown mode: {0}".format(mode),file=sys.stderr)
    print("Accepted modes are: {0}".format(MODES.keys()), file=sys.stderr)
    sys.exit(-1)

if __name__ == '__main__':
  main()


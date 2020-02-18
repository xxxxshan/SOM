import numpy as np

def vote(result):
    t = np.array(result)
    data = t.tolist()
    label = result['label']
    labels = list(set(label))
    print(labels)
    count = 0
    for row, l in zip(data, label):
        r = []
        for a in row[:-1]:
            p = np.argmax(a)
            r.append(labels[p])

        rr = np.argmax(np.bincount(r))
        print('list:{0} vote:{1} label:{2}'.format(r,l,rr))
        if rr == l:
            count += 1
    accurary = count/len(label)
    print("most vote: ",accurary)
    return accurary
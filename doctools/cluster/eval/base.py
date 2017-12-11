from collections import Counter

def purity(clusters, classes):
    def _correct(cluster, labels):
        # Find maximum occuring element in cluster
        frequencies = Counter(clusters)
        matching, frequency = max(frequencies.items(), key=lambda x: x[1])

        # Return how many got classified accurately.
        accurate = Counter(labels)[matching]

    accurate = map(_correct, clusters, classes)
    count = sum(map(len, clusters))

    return accurate/count


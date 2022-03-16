import numpy as np


def convert_labels(labels): # konwersja danych wyjsciowych dla zbiorów umozliwiajaca obliczanie bledow perceptronu oraz latwe porownywanie wynikow
    y_labels = None
    for label in labels:
        number = [0. for i in range(10)]
        number[int(label)] = 1.
        number = np.array(number)
        if y_labels is None:
            y_labels = number
        else:
            y_labels = np.vstack([y_labels, number])

    return y_labels


def split_set_to_validate(our_set, k): # rozdzial podanego zbioru na odpowiednie zbiory przy walidacji
    splitted_set = []
    cutoff_points = []
    elem_to_split = int(len(our_set) / k)
    for i in range(k):
        splitted_set.append(our_set[elem_to_split * i:elem_to_split * (i + 1)])
        cutoff_points.append((elem_to_split*i, elem_to_split*(i + 1)))
    return splitted_set, cutoff_points


def cut_set(our_set, start, end):
    indexes_to_del = [i for i in range(start, end)]

    return np.delete(our_set, indexes_to_del)    
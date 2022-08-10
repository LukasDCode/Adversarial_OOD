import numpy as np
from utils.calculate_softmax_score import calculate_softmax_score
from utils.read_from_pickle_file import get_cifar10_class_averages
from utils.cifar10_labels import cifar10_labels
from utils.cifar100_labels import cifar100_labels


def print_batch_image_modification_path(best_softmax_list, best_idx, num_classes, original_label_batch):
    batch_size = original_label_batch.size()[0]
    for batch_index in range(batch_size):
        original_label = cifar100_labels[original_label_batch[batch_index]]
        print_single_image_modification_path(best_softmax_list[best_idx[batch_index].item()], num_classes, original_label, batch_index)


def print_single_image_modification_path(best_softmax_list, num_classes, original_label, batch_index=0):
    print("Originally:   ", original_label.upper())
    for iteration_index, iteration in enumerate(best_softmax_list):
        softmax_prediction = calculate_softmax_score(iteration[batch_index], num_classes)
        max_value_list, max_indices_list = get_ranked_values_and_indices(softmax_prediction)
        class_prediction_string_concat = str(iteration_index) + ". iteration - "
        for index in range(num_classes):
            class_prediction_string_concat += cifar10_labels[max_indices_list[index]].upper() + ": " + str(max_value_list[index])
            if index < num_classes-1:
                class_prediction_string_concat += ", "
        print(class_prediction_string_concat)
    avg_score_of_final_detected_class = get_cifar10_class_averages()[max_indices_list[0]]
    print("The avg. score of", cifar10_labels[max_indices_list[0]].upper().upper(), "is",
          avg_score_of_final_detected_class)
    print() # only as a nice separator between different images


def get_max_value_and_index_from_softmax_score(softmax_score_list):
    max_value_return, max_value_index_return = 0, 0
    for softmax in softmax_score_list:
        max_value = np.max(softmax)
        max_value_index = np.argmax(softmax)
        if max_value > max_value_return:
            max_value_index_return = max_value_index
            max_value_return = max_value
    return max_value_return, max_value_index_return


def get_ranked_values_and_indices(softmax_score_list):
    max_indices_list = [i for i in range(len(softmax_score_list[0]))]
    return (list(t) for t in zip(*sorted(zip(softmax_score_list[0], max_indices_list), reverse=True)))


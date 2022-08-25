import pickle

def get_cifar10_class_averages():
    # returns dict with 10 entries, the accuracy of the 10 classes of cifar10 sorted alphabetically
    pickle_file_path = '/home-local/koner/lukas/Adversarial_OOD/data/' #'/home/wiss/koner/Lukas/adversarial_ood/data/'
    pickle_file_name_rel = 'cifar10_test_relative_softmax_averages.pickle'
    with open(pickle_file_path + pickle_file_name_rel, 'rb') as handle:
        pickle_content = pickle.load(handle)
    return pickle_content
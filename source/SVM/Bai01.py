from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(100)
N = 150

def main():
    centers = [[2, 2], [7, 7]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)

    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 

    nhom_0 = []
    nhom_1 = []

    SIZE = train_data.shape[0]
    for i in range(0, SIZE):
        if train_labels[i] == 0:
            nhom_0.append([train_data[i,0], train_data[i,1]])
        elif train_labels[i] == 1:
            nhom_1.append([train_data[i,0], train_data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)


    svc = LinearSVC(C = 100, loss="hinge", random_state=42, max_iter = 100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    predicted = svc.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    print('sai so:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = svc.predict(my_test)

    print('Ket qua nhan dang la nhom:', ket_qua[0])

    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')


    decision_function = svc.decision_function(train_data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = train_data[support_vector_indices]
    support_vectors_x = support_vectors[:,0]
    support_vectors_y = support_vectors[:,1]

    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        svc,
        train_data,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.legend(['Nhom 0', 'Nhom 1'])

    plt.show()    


if __name__ == '__main__':
    main()

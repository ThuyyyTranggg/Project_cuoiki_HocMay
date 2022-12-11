import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

#import source.GiamDanDaoHam.Bai01 as bai1
import source.GiamDanDaoHam.Bai02 as bai2
import source.GiamDanDaoHam.Bai02a as bai2a


st.sidebar.markdown("#Đạo hàm giảm dần")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Menu',["Bài 01", "Bài 02", "Bài 02a", "Bài 03", "Bài 04", "Bài 05", "Temp"]) 
if(app_mode == 'Bài 01'):
    st.title("BÀI 01")
    def grad(x):
        return 2*x+ 5*np.cos(x)
    def cost(x):
        return x**2 + 5*np.sin(x)

    def myGD1(x0, eta):
        x = [x0]
        for it in range(100):
            x_new = x[-1] - eta*grad(x[-1])
            if abs(grad(x_new)) < 1e-3: # just a small number
                break
            x.append(x_new)
        return (x, it)
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost(x)
    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx**2 + 5*np.sin(xx)
    z = [xx, yy]
    fig = plt.subplot(2,4,1)
    plt.plot(xx,yy)
    index = 0
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])



    plt.subplot(2,4,2)
    plt.plot(xx, yy)
    index = 1
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,3)
    plt.plot(xx, yy)
    index = 2
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,4)
    plt.plot(xx, yy)
    index = 3
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,5)
    plt.plot(xx, yy)
    index = 4
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,6)
    plt.plot(xx, yy)
    index = 5
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,7)
    plt.plot(xx, yy)
    index = 7
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,8)
    plt.plot(xx, yy)
    index = 11
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
elif(app_mode == 'Bài 02'):
    st.title("BÀI 02")
    X = np.random.rand(1000)

    y = 4 + 3 * X + .5*np.random.randn(1000)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    x0 = 0
    x1 = 1
    y0 = w*x0 + b
    y1 = w*x1 + b

    z = [x0, x1]
    t = [y0, y1]
    plt.plot(X, y, 'bo', markersize = 2)
    b = plt.plot([x0, x1], [y0, y1], 'r')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
elif(app_mode == 'Bài 02a'):
    st.title("BÀI 02a")
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000) # noise added

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    w, b = model.coef_[0][0], model.intercept_[0]
    sol_sklearn = np.array([b, w])
    print('Solution found by sklearn:', sol_sklearn)

    # Building Xbar 
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)

    st.write("Sol dound by GD: w = ", w1[-1], ',\nafter %d iterations.' %(it1+1))
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))

elif(app_mode == 'Bài 03'):
    st.title('Bài 3')
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f & w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    st.write('Sol found by GD: w = ', w1[-1], ',\tafter %d iterations.' %(it1+1))
    # for item in w1:
    #     st.write(item, cost(item))

    # st.write(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    temp = w1[0]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax = plt.axes(projection="3d")
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[1]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[2]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[3]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)


    ax.plot_wireframe(b, w, z)
    ax.set_xlabel("b")
    ax.set_ylabel("w")

    st.pyplot(fig=None, clear_figure=None)

elif(app_mode == 'Bài 04'):
    st.title("BÀI 04")
    x = np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    plt.contour(X, Y, Z, 10)
    plt.show()
    st.pyplot()
elif(app_mode == 'Bài 05'):
    st.title("BÀI 05")
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    print('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    for item in w1:
        print(item, cost(item))

    print(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    plt.contour(b, w, z, 45)
    bdata = []
    wdata = []
    for item in w1:
        plt.plot(item[0], item[1], 'ro', markersize = 3)
        bdata.append(item[0])
        wdata.append(item[1])

    plt.plot(bdata, wdata, color = 'b')

    plt.xlabel('b')
    plt.ylabel('w')
    plt.axis('square')
    plt.show()
    st.pyplot()
else:
    st.title("TEMP")
    ax = plt.axes(projection="3d")

    X = np.linspace(-2, 2, 21)
    Y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(X, Y)
    Z = X*X + Y*Y
    ax.plot_wireframe(X, Y, Z)
    plt.show()
    st.pyplot()
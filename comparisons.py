import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# ==== target function (real function) ====
def true_function(x):
    return np.sin(x) + 0.3 * x

# ==== training data with noise ====
X_train = np.linspace(0, 10, 10).reshape(-1, 1)
y_train = true_function(X_train).ravel() + np.random.normal(0, 0.2, size=X_train.shape[0])

# ==== test data ====
X_test = np.linspace(0, 10, 300).reshape(-1, 1)

# ==== 1. RBF Kernel GP ====
rbf_kernel = RBF(length_scale=1.0)
gp_rbf = GaussianProcessRegressor(kernel=rbf_kernel, alpha=0.2**2)
gp_rbf.fit(X_train, y_train)
y_rbf, y_rbf_std = gp_rbf.predict(X_test, return_std=True)

# ==== 2. Matérn Kernel GP (ν=1.5) ====
matern_kernel = Matern(length_scale=1.0, nu=1.5)
gp_matern = GaussianProcessRegressor(kernel=matern_kernel, alpha=0.2**2)
gp_matern.fit(X_train, y_train)
y_matern, y_matern_std = gp_matern.predict(X_test, return_std=True)

# ==== Figure 1: RBF ====
plt.figure(figsize=(10, 5))
plt.plot(X_test, true_function(X_test), 'k--', label="True Function")
plt.scatter(X_train, y_train, c='black', label="Train Data")
plt.plot(X_test, y_rbf, 'b', label="GP Mean (RBF)")
plt.fill_between(X_test.ravel(), y_rbf - y_rbf_std, y_rbf + y_rbf_std, color='blue', alpha=0.2, label="±1 std")
plt.title("GP Regression with RBF Kernel")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Figure 2: Matérn ====
plt.figure(figsize=(10, 5))
plt.plot(X_test, true_function(X_test), 'k--', label="True Function")
plt.scatter(X_train, y_train, c='black', label="Train Data")
plt.plot(X_test, y_matern, 'g', label="GP Mean (Matérn ν=1.5)")
plt.fill_between(X_test.ravel(), y_matern - y_matern_std, y_matern + y_matern_std, color='green', alpha=0.2, label="±1 std")
plt.title("GP Regression with Matérn Kernel (ν=1.5)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Multilayer Perceptron
小技巧：numpy对象的"序列化"
```python
# 保存为npz格式对象，极大减小文件体积
np.savez_compressed('mnist_scaled.npz',X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
# 返回的字典格式
mnist = np.load('mnist_scaled.npz')
X_train,y_train,X_test,y_test = [mnist[f] for f in mnist.files]
```
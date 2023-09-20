import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

input = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
output = np.array([0.0, 1.0, 1.0, 0.0])

X = torch.tensor(input, dtype=torch.float32)
Y = torch.tensor(output, dtype=torch.float32).reshape(-1, 1)

n_hidden = 2

model = nn.Sequential(nn.Linear(2, n_hidden),
	nn.Sigmoid(),
	nn.Linear(n_hidden, 1),
	nn.Sigmoid())

y_i = model(X)

loss_fn = nn.MSELoss()  # Mean Square Error
optimizer = optim.Adam(model.parameters(), lr=0.1)

n_epochs = 100

for epoch in range(n_epochs):
	y_pred = model(X)
	loss = loss_fn(y_pred, Y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(f'Finished epoch {epoch}, latest loss {loss}')

print('Arquitectura de la red:')
print(model)

print('Salida antes de entrenar:')
print(y_i)

print('Salida después de entrenar:')
print(model(X))

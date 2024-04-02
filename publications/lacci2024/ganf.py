from mtsa.models.ganf import GANF

sampling_rate=50

model_GANF = GANF(sampling_rate=sampling_rate)

epochs_max = 500

for epoch in range(epochs_max):
    model_GANF.fit(X_train, y_train)

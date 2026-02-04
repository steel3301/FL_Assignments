import numpy as np

class Client:
    def __init__(self, client_id, data, labels, lr=0.01):
        self.client_id = client_id
        self. data = data
        self.labels = labels
        self.lr = lr
        self.weights = np.random.randn(data.shape[1])
    
    def train_local(self, epochs=5):

        for _ in range(epochs):
            predictions = self.data.dot(self.weights)
            errors = predictions - self.labels
            gradient = self.data.T.dot(errors) / len(self.data)
            self.weights -= self.lr * gradient
        
        print(f"Client {self.client_id} finished local training")
        return self.weights
    



class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_weights = None

    def aggregate(self, client_weights):

        self.global_weights = np.mean(client_weights, axis = 0)
        print("\nServer aggregated weights using Federated Averaging...\n")
        return self.global_weights
    

    def send_weights(self):
        return self.global_weights
    


def generate_client_data(num_clients=3, samples_per_client=30):
    
    clients = []
    for i in range(num_clients):
        x = np.random.rand(samples_per_client, 1)
        y = 2 * x.squeeze() + np.random.randn(samples_per_client) * 0.1
        clients.append((x, y))
    return clients


def federated_learning_simulation(num_clients=3, rounds=5):
    server = Server(num_clients)

    client_data = generate_client_data(num_clients)

    clients = [
        Client(i, data=x, labels=y) for i, (x, y) in enumerate(client_data)
    ]

    server.global_weights = np.zeros(clients[0].weights.shape)

    for r in range(rounds):
        print(f"\n=========================================Federated Average for Round {r+1}=========================================\n")

        client_weights = []

        for client in clients:
            client.weights = server.send_weights().copy()

        
        for client in clients:
            w = client.train_local()
            client_weights.append(w)

        # SERVER aggregates client updates
        global_w = server.aggregate(client_weights)

        print(f"Updated Global Weights: {global_w}\n")

    print("\nTraining completed!")
    print(f"Final Global Model Weights: {server.global_weights}")


if __name__ == "__main__":
    federated_learning_simulation()